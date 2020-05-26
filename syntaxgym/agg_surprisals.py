import argparse
from copy import deepcopy
import logging
from pathlib import Path

import pandas as pd

from syntaxgym import utils
from syntaxgym.suite import Sentence, Region

L = logging.getLogger(__name__)


def aggregate_surprisals(surprisals, tokens, unks, suite, spec):
    # check that specified metrics are implemented in utils.METRICS
    metrics = suite.meta["metric"]
    if metrics == 'all':
        metrics = utils.METRICS.keys()
    else:
        # if only one metric specified, convert to singleton list
        metrics = [metrics] if type(metrics) == str else metrics
    utils.validate_metrics(metrics)

    ret = deepcopy(suite)
    surprisals = surprisals.reset_index().set_index("sentence_id")
    sent_idx = 0

    # iterate through surprisal file, matching tokens with regions
    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item['conditions']):
            # fetch sentence data
            sent_tokens = tokens[sent_idx]
            sent_unks = unks[sent_idx]
            sent_surps = surprisals.loc[sent_idx + 1]
            t_idx = 0

            if len(sent_tokens) != len(sent_unks) or len(sent_unks) != len(sent_surps):
                L.debug("%s", sent_tokens)
                L.debug("%s", sent_unks)
                L.debug("%s", sent_surps)
                raise ValueError("Mismatched lengths between tokens, unks, and surprisals")
            elif sent_tokens != list(sent_surps.token):
                raise ValueError("Mismatched tokens between tokens and surprisals data frame")

            # Checks done -- strip down data a bit
            sent_surps = sent_surps.surprisal.values

            # initialize Sentence object for current sentence
            sent = Sentence(spec, sent_tokens, sent_unks, item_num=i_idx+1, **cond)

            # iterate through regions in sentence
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
    ret.meta['model'] = spec['name']
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
