from functools import lru_cache
import json
from pathlib import Path
import sys

import jsonschema
import requests

import nose
from nose.tools import *

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)
L = logging.getLogger(__name__)

from suite import Sentence

sys.path.append(str(Path(__file__).parent))
from harness import LM_ZOO_IMAGES, run_image_command_get_stdout


SPEC_SCHEMA_URL = "https://cpllab.github.io/lm-zoo/schemas/language_model_spec.json"

DUMMY_SPEC_PATH = Path(__file__).parent / "dummy_specs"



##################################

@lru_cache(maxsize=None)
def _get_spec(image, tag=None):
    return json.loads(run_image_command_get_stdout(image, "spec", tag=tag))


def _test_individual_spec(image, schema, tag=None):
    print(f"{image}:{tag}")
    jsonschema.validate(instance=_get_spec(image, tag=tag), schema=schema)

def test_specs():
    """
    Validate specs against the lm-zoo standard.
    """
    schema_json = requests.get(SPEC_SCHEMA_URL).json()
    for image, tag in LM_ZOO_IMAGES:
        yield _test_individual_spec, image, schema_json, tag


def test_eos_sos():
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "test."}
    ]
    spec = _get_spec("lmzoo-basic-eos-sos")
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
    spec = _get_spec("lmzoo-basic")
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
    spec = _get_spec("lmzoo-basic")
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
    spec = _get_spec("lmzoo-basic")
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
    spec = _get_spec("lmzoo-basic")
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
    spec = _get_spec("lmzoo-basic")
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
    spec = _get_spec("lmzoo-basic")
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
    spec = _get_spec("lmzoo-basic-uncased")
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
    spec = _get_spec("lmzoo-basic-nopunct")
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
