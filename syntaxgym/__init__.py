import json

from lm_zoo import spec, tokenize, unkify, get_surprisals

from syntaxgym import utils
from syntaxgym.agg_surprisals import aggregate_surprisals
from syntaxgym.get_sentences import get_sentences

__version__ = "0.1"


def compute_surprisals(model_name, suite_file):
    if not hasattr(suite_file, "read"):
        suite_file = open(suite_file, "r")

    image_spec = spec(model_name)
    suite = json.load(suite_file)

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
