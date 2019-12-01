import utils
import argparse
import pandas as pd
from pathlib import Path
from suite import Sentence, Region

class TokenMismatch(Exception):
    def __init__(self, token1, token2, t_idx):
        msg = '''
tokens \"%s\" and \"%s\" do not match (line %d in surprisal file)
        ''' % (token1, token2, t_idx)
        Exception.__init__(self, msg)

def agg_surprisals(surprisals, tokens, unks, in_data, model):
    # check that specified metrics are implemented in utils.METRICS
    metrics = in_data['meta']['metric']
    if metrics == 'all':
        metrics = utils.METRICS.keys()
    else:
        # if only one metric specified, convert to singleton list
        metrics = [metrics] if type(metrics) == str else metrics
    utils.validate_metrics(metrics)

    # store tokens and surprisal values from surprisal file
    TOKENS = surprisals['token'].values
    SURPRISALS = surprisals['surprisal'].values

    # initialize counters for token and sentence from in_data
    t_idx, s_idx = 0, 0

    # iterate through surprisal file, matching tokens with regions
    for i_idx, item in enumerate(in_data['items']):
        for c_idx, cond in enumerate(item['conditions']):
            # sent_tokens must match TOKENS exactly
            sent_tokens = tokens[s_idx]
            sent_unks = unks[s_idx]
            sent = Sentence(**cond, tokens=sent_tokens, unks=sent_unks, model=model)
            for r_idx, region in enumerate(sent.regions):
                for token in region.tokens:
                    # find matching surprisal value for token
                    if token == TOKENS[t_idx]:
                        region.token_surprisals.append(SURPRISALS[t_idx])
                        t_idx += 1
                    else:
                        raise TokenMismatch(token, TOKENS[t_idx], t_idx + 2)
                # get dictionary of aggregate surprisal values for each metric
                vals = {m : region.agg_surprisal(m) for m in metrics}
                # add to original dict
                in_data['items'][i_idx]['conditions'][c_idx]['regions'][r_idx]['metric_value'] = vals
            # update sentence counter
            s_idx += 1
    in_data['meta']['model'] = model
    return in_data

def main(args):
    # read input test suite and token-level surprisals
    in_data = utils.load_json(args.i)
    surprisals = pd.read_csv(args.surprisal, delim_whitespace=True)

    # obtain tokens and unk mask for sentences
    tokens = utils.tokenize_file(args.sentences, args.image)
    unks = utils.unkify_file(args.sentences, args.image)

    # aggregate token-level --> region-level surprisals
    out_data = agg_surprisals(surprisals, tokens, unks, in_data, args.model)
    utils.write_json(out_data, args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate surprisal')
    parser.add_argument('--surprisal', '-surprisal', type=Path,
                        help='path to file containing token-based surprisals')
    parser.add_argument('--sentences', '-sentences', type=Path,
                        help='path to file containing sentences')
    parser.add_argument('--model', '-m', type=str,
                        help='model name')
    parser.add_argument('--image', '-image', type=str, help='Docker image name')
    parser.add_argument('--i', '-i', type=Path,
                        help='path to JSON file with input data')
    parser.add_argument('--o', '-o', type=Path,
                        help='path to JSON file to write output data')
    args = parser.parse_args()
    main(args)
