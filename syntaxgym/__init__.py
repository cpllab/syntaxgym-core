import json

from lm_zoo import spec, tokenize, unkify, get_surprisals

from syntaxgym import utils
from syntaxgym.agg_surprisals import aggregate_surprisals
from syntaxgym.get_sentences import get_sentences
from syntaxgym.suite import Suite

__version__ = "0.1"


def _load_suite(suite_ref):
    if not isinstance(suite_ref, dict):
        if not hasattr(suite_ref, "read"):
            suite_ref = open(suite, "r")
        suite = json.load(suite_ref)
    else:
        suite = suite_ref

    return Suite.from_dict(suite)


def compute_surprisals(model_name, suite):
    """
    Compute per-region surprisals for a language model on the given suite.

    Args:
        model_name: Reference to an LM Zoo model (either a model in the
            registry or a Docker image reference)
        suite_file: A path or open file stream to a suite JSON file, or an
            already loaded suite dict

    Returns:
        An evaluated test suite dict --- a copy of the data from
        ``suite_file``, now including per-region surprisal data
    """
    suite = _load_suite(suite)
    image_spec = spec(model_name)

    # Convert to sentences
    suite_sentences = get_sentences(suite)

    # First compute surprisals
    surprisals_df = get_surprisals(model_name, suite_sentences)

    # Track tokens+unks
    tokens = tokenize(model_name, suite_sentences)
    unks = unkify(model_name, suite_sentences)

    # Now aggregate over regions and get result df
    result = aggregate_surprisals(surprisals_df, tokens, unks, suite, image_spec)

    return result


def evaluate(suite):
    """
    Evaluate prediction results on the given suite. The suite must contain
    surprisal estimates for all regions.
    """
    suite = _load_suite(suite)
