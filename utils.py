import json
import numpy as np
from inspect import getargspec
import subprocess

METRICS = {
    'sum': sum,
    'mean': np.mean,
    'median': np.median,
    'range': np.ptp,
    'max': max,
    'min': min
}

MODELS = ['grnn', 'transformer-xl', 'rnng', 'jrnn']

def flatten(l):
    """
    Flattens a list. Credit to https://stackoverflow.com/a/952952
    """
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

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
    # if only one metric specified, convert to list
    metrics = [metrics] if len(metrics) == 1 else metrics
    if any(m not in METRICS for m in metrics):
        bad_metrics = [m for m in metrics if m not in METRICS]
        raise ValueError('Unknown metrics: {}'.format(bad_metrics))

def run(cmd_str):
    """
    Runs specified command using subprocess and returns list of lines
    from stdout.
    """
    res = subprocess.run(cmd_str.split(), stdout=subprocess.PIPE)
    return res.stdout.decode('utf-8').split('\n')

def tokenize_file(sentence_path, model):
    """
    Tokenizes file at sentence_path according to model.
    If model is None, then split tokens based on whitespace.
    """
    if model is None:
        with open(sentence_path, 'r') as f:
            sentences = f.readlines()
    else:
        # need to call external script to avoid hanging PIPE
        cmd = './tokenize_file %s %s' % (model, sentence_path)
        sentences = run(cmd)
    tokens = [s.strip().split(' ') for s in sentences]
    return tokens
    
def unkify_file(sentence_path, model):
    """
    Unkifies file at sentence_path according to model.
    If model is None, then return no unks (all 0s).
    Note that the returned list is flattened over sentences.
    """
    if model is None:
        tokens = tokenize_file(sentence_path, model)
        unks = [0 for sentence in tokens for t in sentence]
    else:
        # need to call external script to avoid hanging PIPE
        cmd = './unkify_file %s %s' % (model, sentence_path)
        unk_mask = run(cmd)
        unk_mask = [u for u in unk_mask if u != '']
        unks = [int(s) for u in unk_mask for s in u.split(' ')]
    return unks