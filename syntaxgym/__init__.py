import json
from pathlib import Path
from typing import Union, Dict, TextIO

from lm_zoo import get_registry, spec, tokenize, unkify, get_surprisals
from lm_zoo.models import Model, HuggingFaceModel
import pandas as pd

from syntaxgym import utils
from syntaxgym.agg_surprisals import aggregate_surprisals
from syntaxgym.suite import Suite

__version__ = "0.8a1"


def _load_suite(suite_ref: Union[str, Path, TextIO, Dict, Suite]) -> Suite:
    if isinstance(suite_ref, Suite):
        return suite_ref

    # Load from dict / JSON file / JSON path
    if not isinstance(suite_ref, dict):
        if not hasattr(suite_ref, "read"):
            suite_ref = open(suite_ref, "r")
        suite = json.load(suite_ref)
    else:
        suite = suite_ref
    return Suite.from_dict(suite)


def compute_surprisals(model: Model, suite) -> Suite:
    """
    Compute per-region surprisals for a language model on the given suite.

    Args:
        model: An LM Zoo ``Model``.
        suite_file: A path or open file stream to a suite JSON file, or an
            already loaded suite dict

    Returns:
        An evaluated test suite dict --- a copy of the data from
        ``suite_file``, now including per-region surprisal data
    """
    suite = _load_suite(suite)

    # Convert to sentences
    suite_sentences = list(suite.iter_sentences())

    # First compute surprisals
    surprisals_df = get_surprisals(model, suite_sentences)

    # Track tokens
    tokens = tokenize(model, suite_sentences)

    # Now aggregate over regions and get result df
    result = aggregate_surprisals(model, surprisals_df, tokens, suite)

    return result


def evaluate(suite, return_df=True):
    """
    Evaluate prediction results on the given suite. The suite must contain
    surprisal estimates for all regions.
    """
    suite = _load_suite(suite)
    results = suite.evaluate_predictions()
    if not return_df:
        return suite, results

    # Make a nice dataframe
    results_data = [(suite.meta["name"], pred.idx, item_number, result)
                    for item_number, preds in results.items()
                    for pred, result in preds.items()]
    return pd.DataFrame(results_data, columns=["suite", "prediction_id", "item_number", "result"]) \
            .set_index(["suite", "prediction_id", "item_number"])
