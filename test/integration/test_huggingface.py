"""
Test support for HuggingFace models.
"""

import numpy as np
import pytest

import lm_zoo as Z

from syntaxgym import compute_surprisals, evaluate
from syntaxgym.agg_surprisals import prepare_sentences, prepare_sentences_huggingface
from syntaxgym.suite import Suite

zoo = Z.get_registry()


def huggingface_model_fixture(request):
    """
    Defines a generic HF model fixture to be parameterized in a few different
    ways
    """
    model_ref = request.param
    model = zoo[f"huggingface://{model_ref}"]
    return model


huggingface_model_word_refs = [
    "hf-internal-testing/tiny-random-transfo-xl"
]
"""Word-level tokenization HF models"""


huggingface_model_subword_refs = [
    "hf-internal-testing/tiny-xlm-roberta",
    "hf-internal-testing/tiny-random-gpt_neo",
    "hf-internal-testing/tiny-random-reformer",
]
"""Subword-tokenization HF models"""


huggingface_model = pytest.fixture(
    huggingface_model_fixture,
    scope="module",
    params=huggingface_model_word_refs + huggingface_model_subword_refs)


def _assert_suite_results_equivalent(s1, s2, metric="sum"):
    for i1, i2 in zip(s1.items, s2.items):
        for c1, c2 in zip(i1["conditions"], i2["conditions"]):
            for r1, r2 in zip(c1["regions"], c2["regions"]):
                np.testing.assert_almost_equal(
                    r1["metric_value"][metric],
                    r2["metric_value"][metric],
                    err_msg=(f"{i1['item_number']} "
                             f"{c1['condition_name']} "
                             f"{r1['region_number']}"))


def test_hf_deterministic(dummy_suite_json, huggingface_model):
    """
    Test that suite evaluations are deterministic across multiple invocations.
    """

    suite = Suite.from_dict(dummy_suite_json)
    s1 = compute_surprisals(huggingface_model, suite)

    # Again!
    s2 = compute_surprisals(huggingface_model, suite)

    _assert_suite_results_equivalent(s1, s2)


def test_huggingface_sentences(dummy_suite_json, huggingface_model):
    """
    Test that the two sentence detokenization algorithms agree on results.
    """
    if not huggingface_model.provides_token_offsets:
        pytest.skip("only relevant for models which support HF agg_surprisals")

    suite = Suite.from_dict(dummy_suite_json)
    sentences = list(suite.iter_sentences())

    tokens = Z.tokenize(huggingface_model, sentences)

    default_results = prepare_sentences(
        huggingface_model, tokens, suite)
    hf_results = prepare_sentences_huggingface(
        huggingface_model, tokens, suite)

    assert default_results == hf_results
