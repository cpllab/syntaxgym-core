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
import docker.tls

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
    "gpt_tokenization": "lmzoo-gpt-tokenization",
}
LM_ZOO_IMAGE_TO_DIRECTORY = {image: directory
                             for directory, image in LM_ZOO_IMAGES_TO_BUILD.items()}

# Images to test -- harness images and, optionally, real images
LM_ZOO_IMAGES = []
LM_ZOO_IMAGES.extend((image, "latest") for image in LM_ZOO_IMAGES_TO_BUILD.values())

BUILT_IMAGES = []


def _get_client():
    environment = os.environ

    host = environment.get('DOCKER_HOST')

    # empty string for cert path is the same as unset.
    cert_path = environment.get('DOCKER_CERT_PATH') or None

    # empty string for tls verify counts as "false".
    # Any value or 'unset' counts as true.
    tls_verify = environment.get('DOCKER_TLS_VERIFY')
    if tls_verify == '':
        tls_verify = False
    else:
        tls_verify = tls_verify is not None
    enable_tls = cert_path or tls_verify

    params = {}

    if host:
        params['base_url'] = host

    if enable_tls:
        if not cert_path:
            cert_path = os.path.join(os.path.expanduser('~'), '.docker')

        if not tls_verify and assert_hostname is None:
            # assert_hostname is a subset of TLS verification,
            # so if it's not set already then set it to false.
            assert_hostname = False

        params['tls'] = docker.tls.TLSConfig(
            client_cert=(os.path.join(cert_path, 'cert.pem'),
                        os.path.join(cert_path, 'key.pem')),
            ca_cert=os.path.join(cert_path, 'ca.pem'),
            verify=tls_verify,
        )

    return docker.APIClient(**params)


@lru_cache(maxsize=None)
def image_spec(image, tag=None):
    return json.loads(run_image_command_get_stdout(image, "spec", tag=tag))

def image_tokenize(image, content, tag=None):
    fd, fpath = tempfile.mkstemp()
    fpath = Path(fpath)

    if not content.endswith("\n"):
        # Images read line-by-line; make sure the last line doesn't get
        # dropped.
        content += "\n"

    os.write(fd, content.encode("utf-8"))
    os.close(fd)

    host_dir = fpath.parent
    # OS X fix: `/var` can't be mounted; mount `/private/var` instead
    if str(host_dir).startswith("/var"):
        host_dir = host_dir.resolve()
    guest_path = Path("/tmp/host") / fpath.name
    ret = run_image_command_get_stdout(image, f"tokenize {guest_path}", tag=tag,
                                       mounts=[(host_dir, "/tmp/host", "ro")])

    os.remove(str(fpath))

    return ret.strip().split(" ")


def build_image(image, tag="latest"):
    try:
        image_dir = LM_ZOO_IMAGE_TO_DIRECTORY[image]
    except KeyError:
        print("Image %s not found in dummy images directory. Skipping." % (image,), file=sys.stderr)
        return

    image_dir = Path(__file__).parent / "dummy_images" / image_dir

    client = _get_client()
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

    client = _get_client()

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

