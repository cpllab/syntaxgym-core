"""
Supporting code for running SyntaxGym CLI tests.
"""

from functools import lru_cache
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile

import docker

# Silence a few loud modules
logging.getLogger("docker").setLevel(logging.ERROR)
import urllib3
logging.getLogger("urllib3").setLevel(logging.ERROR)


LM_ZOO_IMAGES_TO_BUILD = {
    "basic": "lmzoo-basic",
    "basic_eos_sos": "lmzoo-basic-eos-sos",
    "basic_uncased": "lmzoo-basic-uncased",
    "basic_nopunct": "lmzoo-basic-nopunct",
    "bert_tokenization": "lmzoo-bert-tokenization",
}
LM_ZOO_IMAGE_TO_DIRECTORY = {image: directory
                             for directory, image in LM_ZOO_IMAGES_TO_BUILD.items()}

# Images to test -- harness images and, optionally, real images
LM_ZOO_IMAGES = []
LM_ZOO_IMAGES.extend((image, "latest") for image in LM_ZOO_IMAGES_TO_BUILD.values())

BUILT_IMAGES = []


@lru_cache(maxsize=None)
def image_spec(image, tag=None):
    return json.loads(run_image_command_get_stdout(image, "spec", tag=tag))

def image_tokenize(image, content, tag=None):
    fd, fpath = tempfile.mkstemp()
    os.write(fd, content.encode("utf-8"))
    os.close(fd)

    guest_path = Path("/tmp/host") / os.path.basename(fpath)

    ret = run_image_command_get_stdout(image, f"tokenize {guest_path}", tag=tag,
                                       mounts=[("/tmp", "/tmp/host", "ro")])

    os.remove(fpath)

    return ret.strip().split(" ")


def build_image(image, tag="latest"):
    try:
        image_dir = LM_ZOO_IMAGE_TO_DIRECTORY[image]
    except KeyError:
        print("Image %s not found in dummy images directory." % image, file=sys.stderr)
        raise

    image_dir = Path(__file__).parent / "dummy_images" / image_dir

    client = docker.APIClient()
    out = client.build(path=str(image_dir), rm=True, tag=f"{image}:{tag}")

    BUILT_IMAGES.append(f"{image}:{tag}")
    ret = list(out)

    return ret


def run_image_command(image, command_str, tag=None, pull=False,
                      stdin=None, stdout=sys.stdout, stderr=sys.stderr,
                      mounts=None):
    """
    Run the given shell command inside a Docker container instantiating the
    given image, and stream the output.
    """
    if tag is None:
        tag = "latest"

    if f"{image}:{tag}" not in BUILT_IMAGES:
        build_image(image, tag)

    client = docker.APIClient()

    if pull:
        # First pull the image.
        L.info("Pulling latest Docker image for %s:%s." % (image, tag))
        try:
            image_ret = client.pull(f"{image}", tag=tag)
        except docker.errors.NotFound:
            raise RuntimeError("Image not found.")

    if mounts is None:
        mounts = []

    container = client.create_container(f"{image}:{tag}",

                                        volumes=[guest for _, guest, _ in mounts],
                                        host_config=client.create_host_config(binds={
                                            host: {"bind": guest,
                                                   "mode": mode}
                                            for host, guest, mode in mounts
                                            }),

                                        stdin_open=True,
                                        command=command_str)
    client.start(container)

    if stdin is not None:
        # Send file contents to stdin of container.
        in_stream = client.attach_socket(container, params={"stdin": 1, "stream": 1})
        in_stream._writing = True
        in_stream.write(stdin.read())
        in_stream.close()

    # Stop container and collect results.
    client.stop(container)

    # Collect output.
    container_stdout = client.logs(container, stdout=True, stderr=False)
    container_stderr = client.logs(container, stdout=False, stderr=True)
    if isinstance(container_stdout, bytes):
        container_stdout = container_stdout.decode("utf-8")
    if isinstance(container_stderr, bytes):
        container_stderr = container_stderr.decode("utf-8")

    client.remove_container(container)
    stdout.write(container_stdout)
    stderr.write(container_stderr)


def run_image_command_get_stdout(*args, **kwargs):
    stdout = StringIO()
    kwargs["stdout"] = stdout
    run_image_command(*args, **kwargs)
    return stdout.getvalue()

