"""
Test support for HuggingFace models.
"""

import numpy as np
import pytest

import lm_zoo as Z

from syntaxgym import compute_surprisals, evaluate
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
    "hf-internal-testing/tiny-xlm-roberta"
]
"""Subword-tokenization HF models"""


huggingface_model = pytest.fixture(
    huggingface_model_fixture,
    scope="module",
    params=huggingface_model_word_refs + huggingface_model_subword_refs)


def test_hf_deterministic(dummy_suite_json, huggingface_model):
    """
    Test that suite evaluations are deterministic across multiple invocations.
    """

    suite = Suite.from_dict(dummy_suite_json)
    surps_df = compute_surprisals(huggingface_model, suite)

    # Again!
    surps_df2 = compute_surprisals(huggingface_model, suite)

    for i1, i2 in zip(surps_df.items, surps_df2.items):
        for c1, c2 in zip(i1["conditions"], i2["conditions"]):
            for r1, r2 in zip(c1["regions"], c2["regions"]):
                np.testing.assert_almost_equal(r1["metric_value"]["sum"],
                                               r2["metric_value"]["sum"])
