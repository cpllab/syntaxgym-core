from functools import lru_cache
import json
from pathlib import Path
from pprint import pprint
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
from harness import LM_ZOO_IMAGES, image_spec, image_tokenize


SPEC_SCHEMA_URL = "https://cpllab.github.io/lm-zoo/schemas/language_model_spec.json"

DUMMY_SPEC_PATH = Path(__file__).parent / "dummy_specs"



##################################


def _test_individual_spec(image, schema, tag=None):
    print(f"{image}:{tag}")
    jsonschema.validate(instance=image_spec(image, tag=tag), schema=schema)

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
    spec = image_spec("lmzoo-basic-eos-sos")
    tokens = "<s> This is a test . </s>".split()
    unks = [0, 0, 0, 0, 0, 0, 0]
    sentence = Sentence(spec, tokens, unks, regions=regions)
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
    spec = image_spec("lmzoo-basic")
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
    spec = image_spec("lmzoo-basic")
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
    spec = image_spec("lmzoo-basic")
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
    spec = image_spec("lmzoo-basic")
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


DYNAMIC_CASES = [

    ("Test empty regions",
     ["lmzoo-basic"],
     ["", "This", "is", "", "a test.", ""],
     None,
     {1: [], 2: ["This"], 3: ["is"], 4: [], 5: ["a", "test", "."], 6: []},
     {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}),

    ("Test punctuation-only regions",
     ["lmzoo-basic"],
     "This is , a test .".split(" "),
     None,
     {1: ["This"], 2: ["is"], 3: [","], 4: ["a"], 5: ["test"], 6: ["."]},
     {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}),

    ("Test with uncased image",
     ["lmzoo-basic-uncased"],
     "This is a test.".split(" "),
     None,
     {1: ["this"], 2: ["is"], 3: ["a"], 4: ["test", "."]},
     {1: [], 2: [], 3: [], 4: []}),

    ("Test with punctuation-dropping image",
     ["lmzoo-basic-nopunct"],
     ["Mr. This", "is", ",", "a ---", "test", "."],
     None,
     {1: ["Mr.", "This"], 2: ["is"], 3: [], 4: ["a"], 5: ["test"], 6: []},
     {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}),

    ("Support BERT-style tokenization",
     ["lmzoo-bert-tokenization"],
     ["This is a test sentence."],
     None,
     {1: ["This", "is", "a", "test", "sen", "##tence", "."]},
     {1: []}),

    ("Support GPT2-style tokenization",
     ["lmzoo-gpt-tokenization"],
     ["This is a test sentence."],
     None,
     {1: ["This", "Ġis", "Ġa", "Ġtest", "Ġsen", "tence", "Ġ."]},
     {1: []}),

]


def _test_dynamic_case(image, regions, tokens, expected_region2tokens, expected_oovs):
    spec = image_spec(image)

    print("Spec:")
    pprint(spec)

    print("\n\nRegions:")
    pprint(regions)

    print("\n\nTokens:")
    print(tokens)

    # TODO do we need unks ?
    sentence = Sentence(spec, tokens, unks=None, regions=regions)

    if expected_region2tokens is not None:
        eq_(sentence.region2tokens, expected_region2tokens)

    if expected_oovs is not None:
        eq_(sentence.oovs, expected_oovs)

def test_dynamic_cases():
    for description, images, regions, tokens, expected_region2tokens, expected_oovs in DYNAMIC_CASES:
        if images is None:
            images = LM_ZOO_IMAGES

        # Preprocess regions list.
        if not isinstance(regions[0], dict):
            regions = [{"region_number": i + 1, "content": region}
                       for i, region in enumerate(regions)]

        # Prepare yielded function with a nice description.
        def this_case_fn(*args, **kwargs):
            return _test_dynamic_case(*args, **kwargs)
        this_case_fn.description = description

        for image in images:
            if tokens is None:
                # Tokenize using image.
                tokens = image_tokenize(image, " ".join(r["content"] for r in regions))

            yield this_case_fn, image, regions, tokens, expected_region2tokens, expected_oovs


# def test_special_types():
#     # TODO: if at region boundary, which region do we associate them with?
#     pass

# def test_bpe():
#     pass
