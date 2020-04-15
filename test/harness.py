"""
Supporting code for running SyntaxGym CLI tests.
"""

from io import StringIO
from pathlib import Path
import sys

import docker


LM_ZOO_IMAGES_TO_BUILD = {
    "basic": "lmzoo-basic",
    "basic_eos_sos": "lmzoo-basic-eos-sos",
    "basic_uncased": "lmzoo-basic-uncased",
}
LM_ZOO_IMAGE_TO_DIRECTORY = {image: directory
                             for directory, image in LM_ZOO_IMAGES_TO_BUILD.items()}

# Images to test -- harness images and, optionally, real images
LM_ZOO_IMAGES = []
LM_ZOO_IMAGES.extend((image, "latest") for image in LM_ZOO_IMAGES_TO_BUILD.values())

BUILT_IMAGES = []


def build_image(image, tag="latest"):
    image_dir = LM_ZOO_IMAGE_TO_DIRECTORY[image]
    image_dir = Path(__file__).parent / "dummy_images" / image_dir

    client = docker.APIClient()
    out = client.build(path=str(image_dir), rm=True, tag=f"{image}:{tag}")

    BUILT_IMAGES.append(f"{image}:{tag}")
    ret = list(out)

    return ret


def run_image_command(image, command_str, tag=None, pull=False,
                      stdin=None, stdout=sys.stdout, stderr=sys.stderr):
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

    container = client.create_container(f"{image}:{tag}", stdin_open=True,
                                        command=command_str)
    client.start(container)

    if stdin is not None:
        # Send file contents to stdin of container.
        in_stream = client.attach_socket(container, params={"stdin": 1, "stream": 1})
        in_stream._sock.send(stdin.read())
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

