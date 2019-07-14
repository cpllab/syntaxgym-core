import utils
import argparse
import pandas as pd
from pathlib import Path
from region import Region

def get_agg_surprisals(surprisals, in_data, tokenizer):
    metrics = in_data['meta']['metric']
    if metrics == ['all']:
        metrics = utils.METRICS.keys()
    cur_token = 0
    for i_idx, item in enumerate(in_data['items']):
        for c_idx, cond in enumerate(item['conditions']):
            for r_idx, r in enumerate(cond['regions']):
                region = Region(**r, tokenizer=tokenizer)
                for token in region.tokens:
                    # find matching surprisal value for token
                    assert token == surprisals['token'].values[cur_token]
                    s = surprisals['surprisal'].values[cur_token]
                    region.token_surprisals.append(s)
                    cur_token += 1
                # get dictionary of aggregate surprisal values for each metric
                vals = {m : region.agg_surprisal(m) for m in metrics}
                # add to original dict
                in_data['items'][i_idx]['conditions'][c_idx]['regions'][r_idx]['metric_value'] = vals
    return in_data

def main(args):
    cols = ['sentence_id', 'token_id', 'token', 'surprisal']
    in_data = utils.load_json(args.i)
    surprisals = pd.read_csv(args.surprisal, delim_whitespace=True, names=cols)
    out_data = get_agg_surprisals(surprisals, in_data, args.tokenizer)
    utils.write_json(out_data, args.o)
    print('Wrote results to %s' % args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate surprisal')
    parser.add_argument('--surprisal', '-surprisal', '--s', '-s', type=Path,
                        help='path to file containing token-based surprisals')
    parser.add_argument('--tokenizer', '-tokenizer', '--t', '-t', type=Path,
                        default=None, help='path to tokenizer binary. if None, '
                                           'then split tokens by spaces')
    parser.add_argument('--i', '-i', type=Path,
                        help='path to JSON file with input data')
    parser.add_argument('--o', '-o', type=Path,
                        help='path to JSON file to write output data')
    args = parser.parse_args()
    main(args)