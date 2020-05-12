"""
Supporting code for running SyntaxGym CLI tests.
"""

from functools import lru_cache, wraps
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile

import pytest

import docker
import docker.tls

# Silence a few loud modules
logging.getLogger("docker").setLevel(logging.ERROR)
import urllib3
logging.getLogger("urllib3").setLevel(logging.ERROR)

from syntaxgym import utils


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


@pytest.fixture(scope="module")
def client():
    return utils._get_docker_client().api


@pytest.fixture(scope="module")
def built_image(client, request):
    ref_fields = request.param.rsplit(":", 1) if isinstance(request.param, str) else request.param
    print(ref_fields)
    if len(ref_fields) == 1:
        image, tag = ref_fields[0], "latest"
    else:
        image, tag = ref_fields

    build_image(client, image, tag)
    return ":".join((image, tag))


def build_image(client, image, tag="latest"):
    try:
        image_dir = LM_ZOO_IMAGE_TO_DIRECTORY[image]
    except KeyError:
        print("Image %s not found in dummy images directory. Skipping." % (image,), file=sys.stderr)
        return

    image_dir = Path(__file__).parent / "dummy_images" / image_dir

    out = client.build(path=str(image_dir), rm=True, tag=f"{image}:{tag}")

    BUILT_IMAGES.append(f"{image}:{tag}")
    ret = list(out)

    return ret


def with_images(*images):
    def decorator_with_images(fn):
        ids = [":".join(image_ref) if not isinstance(image_ref, str) else image_ref
               for image_ref in images]

        @pytest.mark.parametrize("built_image", images, ids=ids,
                                 indirect=["built_image"])
        @wraps(fn)
        def my_testfn(*args, **kwargs):
            return fn(*args, **kwargs)

        return my_testfn
    return decorator_with_images
