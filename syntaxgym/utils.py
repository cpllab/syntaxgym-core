from functools import lru_cache
from io import StringIO
import json
import numpy as np
from inspect import getargspec
from pathlib import Path
import subprocess
import sys

import logging
L = logging.getLogger("syntaxgym")

import docker
import tqdm


METRICS = {
    'sum': sum,
    'mean': np.mean,
    'median': np.median,
    'range': np.ptp,
    'max': max,
    'min': min
}

MODELS = ['grnn', 'transformer-xl', 'rnng', 'jrnn', 'ordered-neurons', 'roberta']

class TokenMismatch(Exception):
    def __init__(self, token1, token2, t_idx):
        msg = '''
tokens \"%s\" and \"%s\" do not match (line %d in surprisal file)
        ''' % (token1, token2, t_idx)
        Exception.__init__(self, msg)

def save_args(values):
    """
    Automatically saves constructor arguments to object.
    Credit to https://stackoverflow.com/a/15484172
    """
    for i in getargspec(values['self'].__init__).args[1:]:
        setattr(values['self'], i, values[i])

def load_json(path):
    """
    Loads Path to JSON file as dictionary.
    """
    with path.open() as f:
        d = json.load(f)
    return d

def write_json(d, path):
    """
    Writes dictionary to JSON file specified by Path.
    """
    with path.open('w') as f:
        json.dump(d, f, indent=4)

def read_lines(path):
    """
    Reads lines from file into list with leading and trailing whitespace removed.
    """
    with path.open() as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

def write_lines(lines, path):
    """
    Writes list to Path, separated by \n.
    """
    with path.open('w') as f:
        for l in lines:
            f.write(str(l) + '\n')

def validate_metrics(metrics):
    """
    Checks if specified metrics are valid. Returns None if check passes,
    else raises ValueError.
    """
    if any(m not in METRICS for m in metrics):
        bad_metrics = [m for m in metrics if m not in METRICS]
        raise ValueError('Unknown metrics: {}'.format(bad_metrics))

def run(cmd_str):
    """
    Runs specified command using subprocess and returns list of lines
    from stdout.
    """
    res = subprocess.run(cmd_str.split(), stdout=subprocess.PIPE)
    return res.stdout.decode('utf-8').strip("\n").split('\n')


@lru_cache()
def _get_docker_client():
    return docker.from_env()

def _update_progress(line, progress_bars):
    """
    Process a progress update line from the Docker API for push/pull
    operations, writing to `progress_bars`.
    """
    # From https://github.com/neuromation/platform-client-python/pull/201/files#diff-2d85e2a65d4d047287bea6267bd3826dR771
    try:
        if "id" in line:
            status = line["status"]
            if status == "Pushed" or status == "Download complete":
                if line["id"] in progress_bars:
                    progress = progress_bars[line["id"]]
                    delta = progress["total"] - progress["current"]
                    if delta < 0:
                        delta = 0
                    progress["progress"].update(delta)
                    progress["progress"].close()
            elif status == "Pushing" or status == "Downloading":
                if line["id"] not in progress_bars:
                    if "progressDetail" in line:
                        progress_details = line["progressDetail"]
                        total_progress = progress_details.get(
                            "total", progress_details.get("current", 1)
                        )
                        if total_progress > 0:
                            progress_bars[line["id"]] = {
                                "progress": tqdm.tqdm(
                                    total=total_progress,
                                    leave=False,
                                    unit="B",
                                    unit_scale=True,
                                ),
                                "current": 0,
                                "total": total_progress,
                            }
                if "progressDetail" in line and "current" in line["progressDetail"]:
                    delta = (
                        line["progressDetail"]["current"]
                        - progress_bars[line["id"]]["current"]
                    )
                    if delta < 0:
                        delta = 0
                    progress_bars[line["id"]]["current"] = line["progressDetail"]["current"]
                    progress_bars[line["id"]]["progress"].update(delta)
    except BaseException:
        pass

def _pull_container(image, tag, registry="docker.io", progress_stream=sys.stderr):
    client = _get_docker_client().api

    # First pull the image.
    L.info("Pulling latest Docker image for %s:%s." % (image, tag), err=True)
    try:
        progress_bars = {}
        for line in client.pull(f"{registry}/{image}", tag=tag, stream=True, decode=True):
            if progress_stream is not None:
                # Write pull progress on the given stream.
                _update_progress(line, progress_bars)
            else:
                pass
    except docker.errors.NotFound:
        raise RuntimeError("Image not found.")


def _run_container(image, command_str, tag=None, pull=False,
                   stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                   progress_stream=sys.stderr):
    """
    Run the given command inside a Docker container.
    """
    client = _get_docker_client().api

    if tag is None:
        if ":" in image:
            image, tag = image.rsplit(":", 1)
        else:
            tag = "latest"

    if pull:
        _pull_container(image, tag, progress_stream=progress_stream)

    container = client.create_container(f"{image}:{tag}", stdin_open=True,
                                        command=command_str)
    client.start(container)

    if stdin is not None:
        # Send file contents to stdin of container.
        in_stream = client.attach_socket(container, params={"stdin": 1, "stream": 1})
        os.write(in_stream._sock.fileno(), stdin.read())
        os.close(in_stream._sock.fileno())

    # Stop container and collect results.
    # TODO parameterize timeout
    client.stop(container, timeout=60)

    # Collect output.
    container_stdout = client.logs(container, stdout=True, stderr=False)
    container_stderr = client.logs(container, stdout=False, stderr=True)

    client.remove_container(container)
    stdout.write(container_stdout.decode("utf-8"))
    stderr.write(container_stderr.decode("utf-8"))

def _run_container_get_stdout(*args, **kwargs):
    out = StringIO()
    kwargs["stdout"] = out
    _run_container(*args, **kwargs)
    return out.getvalue()

def get_spec(image):
    """
    Gets model spec from specified Docker image.
    """
    return json.loads(_run_container_get_stdout(image, "spec"))

def tokenize_file(sentence_path, image):
    """
    Tokenizes file at sentence_path according to specified Docker image.
    If image is None, then split tokens based on whitespace.
    """
    if image is None:
        with open(sentence_path, 'r') as f:
            sentences = f.readlines()
    else:
        # need to call external script to avoid hanging PIPE
        tokenize_path = Path(__file__).absolute().parent.parent / "bin" / "tokenize_file"
        cmd = '%s %s %s' % (tokenize_path, image, sentence_path)
        sentences = run(cmd)
    tokens = [s.strip().split(' ') for s in sentences]
    return tokens

def unkify_file(sentence_path, image):
    """
    Unkifies file at sentence_path according to image.
    If image is None, then return no unks (all 0s).
    """
    if image is None:
        tokens = tokenize_file(sentence_path, image)
        mask = [[0 for t in sent_tokens] for sent_tokens in tokens]
    else:
        # need to call external script to avoid hanging PIPE
        unkify_path = Path(__file__).absolute().parent.parent / "bin" / "unkify_file"
        cmd = './unkify_file %s %s' % (image, sentence_path)
        mask = run(cmd)
        # split on newlines to get sentences
        # print(raw_mask)
        # mask = raw_mask.split('\n')
        mask = [u for u in mask if u != '']
        # split on whitespace and convert strings to ints
        mask = [[int(u) for u in sent_unks.split(' ')] for sent_unks in mask]
    return mask
