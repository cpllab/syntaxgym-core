import json
from pathlib import Path

import jsonschema
import requests

import nose
from nose.tools import *

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from agg_surprisals import aggregate_surprisals
from suite import Sentence


SPEC_SCHEMA_URL = "https://cpllab.github.io/lm-zoo/schemas/language_model_spec.json"

def _test_individual_spec(spec_name, spec, schema):
    print(spec_name)
    jsonschema.validate(instance=spec, schema=schema)

def test_specs():
    """
    Validate the dummy specs against the lm-zoo standard.
    """
    schema_json = requests.get(SPEC_SCHEMA_URL).json()
    for spec_path in Path("dummy_specs").glob("*.json"):
        with spec_path.open("r") as spec_f:
            yield _test_individual_spec, spec_path.name, json.load(spec_f), schema_json


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
    eq_(sentence.oovs, {
        0: [],
        1: ["WEIRDADVERB"],
        2: [],
        3 : ["WEIRDNOUN"]
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
    tokens = "This is a <unk> test <unk> <unk> <unk> .".split()
    unks = [0, 0, 0, 1, 0, 1, 1, 1]
    sentence = Sentence(spec, tokens, unks, regions=regions)
    eq_(sentence.oovs, {
        0: [],
        1: [],
        2: [],
        3 : ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"]
    })


def test_consecutive_unk2():
    """
    Consecutive UNKs are mapped to regions by lookahead -- we look ahead in the
    token string for the next non-unk token, and associate all unks up to that
    token with the current region.
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
    tokens = "This is a <unk> test <unk> <unk> <unk> and some more content .".split()
    unks = [0, 0, 0, 1, 0, 1, 1, 1]
    sentence = Sentence(spec, tokens, unks, regions=regions)
    eq_(sentence.oovs, {
        0: [],
        1: [],
        2: [],
        3 : ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"],
        4: [],
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
