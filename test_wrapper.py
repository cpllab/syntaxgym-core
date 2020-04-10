from io import StringIO
import json
from pathlib import Path
import sys

import docker
import jsonschema
import requests

import nose
from nose.tools import *

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)
L = logging.getLogger(__name__)

from agg_surprisals import aggregate_surprisals
from suite import Sentence


LM_ZOO_IMAGES = [
    ("lmzoo-basic", "latest"),
    ("lmzoo-basic-eos-sos", "latest"),
]

LM_ZOO_IMAGES_TO_BUILD = [
    ("basic", "lmzoo-basic"),
    ("basic_eos_sos", "lmzoo-basic-eos-sos"),
]

SPEC_SCHEMA_URL = "https://cpllab.github.io/lm-zoo/schemas/language_model_spec.json"


def run_image_command(image, command_str, tag=None, pull=False,
                      stdin=None, stdout=sys.stdout, stderr=sys.stderr):
    """
    Run the given shell command inside a container instantiating the given
    image, and stream the output.
    """
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


def setup_module():
    # Build relevant lm-zoo images.
    client = docker.APIClient()
    for directory, target in LM_ZOO_IMAGES_TO_BUILD:
        path = Path(__file__).parent / "dummy_images" / directory

        with (path / "Dockerfile").open("r") as docker_f:
            out = client.build(path=str(path), rm=True,
                         tag=target)
            list(out)

def teardown_module():
    # Remove built images.
    pass


##################################


def _test_individual_spec(image, tag, schema):
    print(f"{image}:{tag}")
    spec = json.loads(run_image_command_get_stdout(image, "spec", tag=tag))
    jsonschema.validate(instance=spec, schema=schema)

def test_specs():
    """
    Validate specs against the lm-zoo standard.
    """
    schema_json = requests.get(SPEC_SCHEMA_URL).json()
    for image, tag in LM_ZOO_IMAGES:
        yield _test_individual_spec, image, tag, schema_json


def test_eos_sos():
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "test."}
    ]
    with open("dummy_specs/eos_sos.json", "r") as f:
        spec = json.load(f)
    tokens = "<s> This is a test . </s>".split()
    unks = [0, 0, 0, 0, 0, 0, 0]
    sentence = Sentence(spec, tokens, unks, regions=regions)
    print(sentence.region2tokens)
    eq_(sentence.region2tokens, {
        1: ["<s>", "This"],
        2: ["is"],
        3: ["a"],
        4 : ["test", ".", "</s>"]
    })


def test_unk():
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is WEIRDADVERB"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDNOUN."}
    ]
    with open("dummy_specs/basic.json", "r") as f:
        spec = json.load(f)
    tokens = "This is <unk> a <unk> .".split()
    unks = [0, 0, 1, 0, 1, 0]
    sentence = Sentence(spec, tokens, unks, regions=regions)

    eq_(sentence.region2tokens, {
        1: ["This"],
        2: ["is", "<unk>"],
        3: ["a"],
        4: ["<unk>", "."],
    })

    eq_(sentence.oovs, {
        1: [],
        2: ["WEIRDADVERB"],
        3: [],
        4: ["WEIRDNOUN"],
    })


def test_consecutive_unk():
    """
    Consecutive UNKs are mapped to regions by lookahead -- we look ahead in the
    token string for the next non-unk token, and associate all unks up to that
    token with the current region.
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDADVERB test WEIRDADJECTIVE WEIRDNOUN."}
    ]
    with open("dummy_specs/basic.json", "r") as f:
        spec = json.load(f)
    tokens = "This is a <unk> test <unk> <unk> .".split()
    unks = [0, 0, 0, 1, 0, 1, 1, 1]
    sentence = Sentence(spec, tokens, unks, regions=regions)

    eq_(sentence.region2tokens, {
        1: ["This"],
        2: ["is"],
        3: ["a"],
        4: ["<unk>", "test", "<unk>", "<unk>", "."],
    })

    eq_(sentence.oovs, {
        1: [],
        2: [],
        3: [],
        4: ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"],
    })


def test_consecutive_unk2():
    """
    consecutive unks in the middle of a region, with non-unks following in the
    same region
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDADVERB test WEIRDADJECTIVE WEIRDNOUN and some more"},
        {"region_number": 5, "content": "content."}
    ]
    with open("dummy_specs/basic.json", "r") as f:
        spec = json.load(f)
    tokens = "This is a <unk> test <unk> <unk> and some more content .".split()
    unks = [0, 0, 0, 1, 0, 1, 1, 1]
    sentence = Sentence(spec, tokens, unks, regions=regions)

    eq_(sentence.region2tokens, {
        1: ["This"],
        2: ["is"],
        3: ["a"],
        4: ["<unk>", "test", "<unk>", "<unk>", "and", "some", "more"],
        5: ["content", "."],
    })

    eq_(sentence.oovs, {
        1: [],
        2: [],
        3: [],
        4: ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"],
        5: [],
    })


def test_consecutive_unk3():
    """
    consecutive unks at the end of a region, with more regions after
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDADVERB test WEIRDADJECTIVE WEIRDNOUN"},
        {"region_number": 5, "content": "and some more content."}
    ]
    with open("dummy_specs/basic.json", "r") as f:
        spec = json.load(f)
    tokens = "This is a <unk> test <unk> <unk> and some more content .".split()
    unks = [0, 0, 0, 1, 0, 1, 1, 1]
    sentence = Sentence(spec, tokens, unks, regions=regions)

    eq_(sentence.region2tokens, {
        1: ["This"],
        2: ["is"],
        3: ["a"],
        4: ["<unk>", "test", "<unk>", "<unk>"],
        5: ["and", "some", "more", "content", "."],
    })

    eq_(sentence.oovs, {
        1: [],
        2: [],
        3: [],
        4: ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"],
        5: [],
    })


def test_empty_region():
    regions = [
        {"region_number": 1, "content": ""},
        {"region_number": 2, "content": "This"},
        {"region_number": 3, "content": "is"},
        {"region_number": 4, "content": ""},
        {"region_number": 5, "content": "a test."},
        {"region_number": 6, "content": ""}
    ]
    with open("dummy_specs/basic.json", "r") as f:
        spec = json.load(f)
    tokens = "This is a test .".split()
    unks = [0, 0, 0, 0, 0]
    sentence = Sentence(spec, tokens, unks, regions=regions)
    eq_(sentence.region2tokens, {
        1: [],
        2: ["This"],
        3: ["is"],
        4: [],
        5: ["a", "test", "."],
        6: []
    })

def test_punct_region():
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": ","},
        {"region_number": 4, "content": "a"},
        {"region_number": 5, "content": "test"},
        {"region_number": 6, "content": "."}
    ]
    with open("dummy_specs/basic.json", "r") as f:
        spec = json.load(f)
    tokens = "This is , a test .".split()
    unks = [0, 0, 0, 0, 0, 0]
    sentence = Sentence(spec, tokens, unks, regions=regions)
    eq_(sentence.region2tokens, {
        1: ["This"],
        2: ["is"],
        3: [","],
        4: ["a"],
        5 : ["test"],
        6: ["."]
    })

def test_uncased():
    """
    Test uncased vocabulary.
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "test."}
    ]
    with open("dummy_specs/basic_uncased.json", "r") as f:
        spec = json.load(f)
    tokens = "this is a test .".split()
    unks = [0, 0, 0, 0, 0]
    sentence = Sentence(spec, tokens, unks, regions=regions)

    eq_(sentence.region2tokens, {
        1: ["this"],
        2: ["is"],
        3: ["a"],
        4 : ["test", "."]
    })

def test_remove_punct():
    regions = [
        {"region_number": 1, "content": "This!"},
        {"region_number": 2, "content": "?is"},
        {"region_number": 3, "content": ","},
        {"region_number": 4, "content": "a ---"},
        {"region_number": 5, "content": "test"},
        {"region_number": 6, "content": "."}
    ]
    with open("dummy_specs/basic_nopunct.json", "r") as f:
        spec = json.load(f)
    tokens = "This is a test".split()
    unks = [0, 0, 0, 0]
    sentence = Sentence(spec, tokens, unks, regions=regions)
    eq_(sentence.region2tokens, {
        1: ["This"],
        2: ["is"],
        3: [],
        4: ["a"],
        5 : ["test"],
        6: []
    })

# def test_special_types():
#     # TODO: if at region boundary, which region do we associate them with?
#     pass

# def test_bpe():
#     pass
