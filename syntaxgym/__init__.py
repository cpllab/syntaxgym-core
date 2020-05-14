import json

from lm_zoo import spec, tokenize, unkify, get_surprisals

from syntaxgym import utils
from syntaxgym.agg_surprisals import aggregate_surprisals
from syntaxgym.get_sentences import get_sentences

__version__ = "0.1"


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
    if not isinstance(suite, dict):
        if not hasattr(suite, "read"):
            suite = open(suite, "r")
        suite = json.load(suite)

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
