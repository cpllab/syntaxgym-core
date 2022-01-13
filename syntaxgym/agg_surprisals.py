"""
Defines methods for aligning LM tokens with suite regions, and converting
raw token-level surprisal outputs from models into suites with computed
region-level surprisals.
"""

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from lm_zoo import spec
from lm_zoo.models import Model, HuggingFaceModel

from syntaxgym import utils
from syntaxgym.suite import Suite, Sentence

L = logging.getLogger(__name__)


def _prepare_metrics(suite: Suite) -> List[str]:
    # check that specified metrics are implemented in utils.METRICS
    metrics = suite.meta["metric"]
    if metrics == 'all':
        metrics = utils.METRICS.keys()
    else:
        # if only one metric specified, convert to singleton list
        metrics = [metrics] if type(metrics) == str else metrics
    utils.validate_metrics(metrics)

    return metrics


def prepare_sentences(model: Model, tokens: List[List[str]],
                      suite: Suite) -> List[Sentence]:
    sent_idx = 0
    ret = []

    # Pre-fetch model spec for aggregation algorithm
    model_spec = spec(model)

    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item['conditions']):
            sent_tokens = tokens[sent_idx]

            sent = Sentence(sent_tokens, item_num=i_idx+1, **cond)

            try:
                sent.compute_region2tokens(model_spec)
            except Exception as e:
                print("Tokens: ", sent_tokens, file=sys.stderr)
                print("Region spec: ", cond["regions"], file=sys.stderr)

                raise ValueError("Error occurred while processing item %i, "
                                 "condition %s. Relevant debug information "
                                 "printed to stderr."
                                 % (item["item_number"],
                                    cond["condition_name"])) from e

            ret.append(sent)

    return ret


def prepare_sentences_huggingface(model: Model, tokens: List[List[str]],
                                  suite: Suite) -> List[Sentence]:
    if not model.provides_token_offsets:
        raise NotImplementedError("Only implemented for Huggingface models "
                                  "which support detokenization.")

    region_edges = list(suite.iter_region_edges())

    # Hack: re-tokenize here in order to detokenize back to character-level
    # offsets.
    sentences = list(suite.iter_sentences())
    encoded = model.tokenizer.batch_encode_plus(
        sentences, add_special_tokens=True, return_offsets_mapping=True)

    ret = []

    sent_idx = 0
    for i_idx, item in enumerate(ret.items):
        for c_idx, cond in enumerate(item["conditions"]):
            # fetch sentence data
            tokens_i = encoded.tokens(sent_idx)

            # len(tokens)-length list of tuples (char_start, char_end)
            token_offsets_i: List[Tuple[int, int]] = \
                encoded["offset_mapping"][sent_idx]

            region_edges_i = region_edges[sent_idx]

            # initialize Sentence object for current sentence
            sent = Sentence(tokens_i, item_num=i_idx + 1, **cond)

            region2tokens = [[]]
            r_cursor, t_cursor = 0, 0
            while t_cursor < len(tokens_i):
                token = tokens_i[t_cursor]
                token_char_start, token_char_end = token_offsets_i[t_cursor]

                region_start = region_edges_i[r_cursor]
                region_end = region_edges_i[r_cursor + 1] \
                    if r_cursor + 1 < len(region_edges_i) else np.inf

                # NB region boundaries are left edges, hence the >= here.
                if token_char_start >= region_end:
                    r_cursor += 1
                    region2tokens.append([])
                    continue

                region2tokens[r_cursor].append(token)
                t_cursor += 1

            for r, r_tokens in zip(sent.regions, region2tokens):
                r.tokens = r_tokens

            ret.append(sent)

    return ret


def aggregate_surprisals(model: Model, surprisals: pd.DataFrame,
                         tokens: List[List[str]], suite: Suite):
    metrics = _prepare_metrics(suite)

    ret = deepcopy(suite)
    surprisals = surprisals.reset_index().set_index("sentence_id")

    # Checks
    sent_idx = 0
    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item['conditions']):
            # fetch sentence data
            sent_tokens = tokens[sent_idx]
            sent_surps = surprisals.loc[sent_idx + 1]

            if sent_tokens != list(sent_surps.token):
                raise ValueError("Mismatched tokens between tokens and surprisals data frame")

    # Run sentence prep procedure -- map tokens in each sentence onto regions
    # of corresponding test trial sentence
    if isinstance(model, HuggingFaceModel) and model.provides_token_offsets:
        sentences = prepare_sentences_huggingface(model, tokens, suite)
    else:
        sentences = prepare_sentences(model, tokens, suite)

    # Strip down data a bit
    sent_surps = sent_surps.surprisal.values

    # Bring in surprisals
    sent_idx = 0
    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item["conditions"]):
            sent = sentences[sent_idx]

            sent_tokens = tokens[sent_idx]
            sent_surps = surprisals.loc[sent_idx + 1].surprisal.values

            # iterate through regions in sentence
            t_idx = 0
            for r_idx, region in enumerate(sent.regions):
                for token in region.tokens:
                    # append to region surprisals if exact token match
                    if token == sent_tokens[t_idx]:
                        region.token_surprisals.append(sent_surps[t_idx])
                        t_idx += 1
                    else:
                        raise utils.TokenMismatch(token, sent_tokens[t_idx], t_idx+2)

                # get dictionary of region-level surprisal values for each metric
                vals = {m : region.agg_surprisal(m) for m in metrics}

                # insert surprisal values into original dict
                ret.items[i_idx]['conditions'][c_idx]['regions'][r_idx]['metric_value'] = vals

                # update original dict with OOV information
                ret.items[i_idx]['conditions'][c_idx]['regions'][r_idx]['oovs'] = \
                  sent.oovs[region.region_number]

            # update sentence counter
            sent_idx += 1

    # update meta information with model name
    ret.meta['model'] = spec(model)['name']
    return ret


def main(args):
    # read input test suite and token-level surprisals
    in_data = utils.load_json(args.input)
    surprisals = pd.read_csv(args.surprisal, delim_whitespace=True)

    # obtain spec for model
    spec = utils.get_spec(args.image)

    # obtain tokens and unk mask for sentences
    tokens = utils.tokenize_file(args.sentences, args.image)
    unks = utils.unkify_file(args.sentences, args.image)

    # aggregate token-level --> region-level surprisals
    out_data = aggregate_surprisals(surprisals, tokens, unks, in_data, spec)
    utils.write_json(out_data, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate surprisal')
    parser.add_argument('--surprisal', type=Path,
                        help='path to file containing token-based surprisals')
    parser.add_argument('--sentences', type=Path,
                        help='path to file containing pre-tokenized sentences')

    parser.add_argument('--image', type=str, help='Docker image name')
    parser.add_argument('--input', type=Path,
                        help='path to JSON file with input data')
    parser.add_argument('--output', '-o', type=Path,
                        help='path to JSON file to write output data')
    args = parser.parse_args()
    main(args)
