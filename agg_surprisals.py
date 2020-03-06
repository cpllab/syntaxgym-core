import utils
import argparse
import pandas as pd
from pathlib import Path
from suite import Sentence, Region

def aggregate_surprisals(surprisals, tokens, unks, in_data, spec):
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
            # grab tokens and unks for current sentence from Docker image output
            sent_tokens = tokens[s_idx]
            sent_unks = unks[s_idx]

            # initialize Sentence object for current sentence
            sent = Sentence(spec, sent_tokens, sent_unks, item_num=i_idx+1, **cond)

            # iterate through regions in sentence
            for r_idx, region in enumerate(sent.regions):
                print(region)
                for token in region.tokens:
                    # append to region surprisals if exact token match
                    if token == TOKENS[t_idx]:
                        region.token_surprisals.append(SURPRISALS[t_idx])
                        t_idx += 1
                    else:
                        raise utils.TokenMismatch(token, TOKENS[t_idx], t_idx+2)

                # get dictionary of region-level surprisal values for each metric
                vals = {m : region.agg_surprisal(m) for m in metrics}

                # insert surprisal values into original dict
                in_data['items'][i_idx]['conditions'][c_idx]['regions'][r_idx]['metric_value'] = vals

            # update sentence counter
            s_idx += 1

    # update meta information with model name
    in_data['meta']['model'] = spec['name']
    return in_data

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
                        help='path to file containing sentences')
    parser.add_argument('--image', type=str, help='Docker image name')
    parser.add_argument('--input', type=Path,
                        help='path to JSON file with input data')
    parser.add_argument('--output', '-o', type=Path,
                        help='path to JSON file to write output data')
    args = parser.parse_args()
    main(args)
