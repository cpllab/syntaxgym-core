import utils
import argparse
import pandas as pd
from pathlib import Path
from region import Sentence, Region

class TokenMismatch(Exception):
    pass

def validate_metrics(metrics):
    # if only one metric specified, convert to list
    metrics = [metrics] if len(metrics) == 1 else metrics
    if any(m not in utils.METRICS for m in metrics):
        bad_metrics = [m for m in metrics if m not in utils.METRICS]
        raise ValueError('Unknown metrics: {}'.format(bad_metrics))

def get_agg_surprisals(surprisals, in_data, model):
    # check that specified metrics are implemented in utils.METRICS
    metrics = in_data['meta']['metric']
    if metrics == 'all':
        metrics = utils.METRICS.keys()
    else:
        validate_metrics(metrics)

    # iterate through surprisal file, matching tokens with regions
    cur_token = 0
    for i_idx, item in enumerate(in_data['items']):
        for c_idx, cond in enumerate(item['conditions']):
            sentence = Sentence(**cond, model=model)
            for r_idx, region in enumerate(sentence.regions):
                for token in region.tokens:
                    # find matching surprisal value for token
                    if token == surprisals['token'].values[cur_token]:
                        s = surprisals['surprisal'].values[cur_token]
                        region.token_surprisals.append(s)
                        cur_token += 1
                    else:
                        raise TokenMismatch('''
tokens \"%s\" and \"%s\" do not match (line %d in surprisal file)
                        ''' % (token, surprisals['token'].values[cur_token], cur_token+2))
                # get dictionary of aggregate surprisal values for each metric
                vals = {m : region.agg_surprisal(m) for m in metrics}
                # add to original dict
                in_data['items'][i_idx]['conditions'][c_idx]['regions'][r_idx]['metric_value'] = vals
    return in_data

def main(args):
    in_data = utils.load_json(args.i)
    surprisals = pd.read_csv(args.surprisal, delim_whitespace=True)
    out_data = get_agg_surprisals(surprisals, in_data, args.model)
    utils.write_json(out_data, args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate surprisal')
    parser.add_argument('--surprisal', '-surprisal', '--s', '-s', type=Path,
                        help='path to file containing token-based surprisals')
    parser.add_argument('--model', '-model', '--m', '-m', type=str,
                        default=None, help='model for cpllab Docker container. '
                                           'if None, then split tokens by spaces')
    parser.add_argument('--i', '-i', type=Path,
                        help='path to JSON file with input data')
    parser.add_argument('--o', '-o', type=Path,
                        help='path to JSON file to write output data')
    args = parser.parse_args()
    main(args)