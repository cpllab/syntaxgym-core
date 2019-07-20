import utils
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from region import Sentence, Region

class TokenMismatch(Exception):
    pass

def tokenize_file(sentence_path, model):
    if model is None:
        with open(sentence_path, 'r') as f:
            sentences = f.readlines()
    else:
        # need to call external script to avoid hanging PIPE
        cmd = './tokenize_file %s %s' % (model, sentence_path)
        sentences = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
        sentences = sentences.stdout.decode('utf-8').split('\n')
    tokens = [s.strip().split(' ') for s in sentences]
    return tokens

def get_agg_surprisals(surprisals, sentence_tokens, in_data, model):
    # check that specified metrics are implemented in utils.METRICS
    metrics = in_data['meta']['metric']
    if metrics == 'all':
        metrics = utils.METRICS.keys()
    else:
        utils.validate_metrics(metrics)

    # initialize counters for token and sentence
    t_idx, s_idx = 0, 0
    # iterate through surprisal file, matching tokens with regions
    for i_idx, item in enumerate(in_data['items']):
        for c_idx, cond in enumerate(item['conditions']):
            sentence = Sentence(**cond, tokens=sentence_tokens[s_idx])
            for r_idx, region in enumerate(sentence.regions):
                for token in region.tokens:
                    # find matching surprisal value for token
                    if token == surprisals['token'].values[t_idx]:
                        s = surprisals['surprisal'].values[t_idx]
                        region.token_surprisals.append(s)
                        t_idx += 1
                    else:
                        raise TokenMismatch('''
tokens \"%s\" and \"%s\" do not match (line %d in surprisal file)
                        ''' % (token, surprisals['token'].values[t_idx], t_idx+2))
                # get dictionary of aggregate surprisal values for each metric
                vals = {m : region.agg_surprisal(m) for m in metrics}
                # add to original dict
                in_data['items'][i_idx]['conditions'][c_idx]['regions'][r_idx]['metric_value'] = vals
            # update sentence counter
            s_idx += 1
    in_data['meta']['model'] = model
    return in_data

def main(args):
    in_data = utils.load_json(args.i)
    surprisals = pd.read_csv(args.surprisal, delim_whitespace=True)
    sentence_tokens = tokenize_file(args.sentences, args.model)
    out_data = get_agg_surprisals(surprisals, sentence_tokens, in_data, args.model)
    utils.write_json(out_data, args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate surprisal')
    parser.add_argument('--surprisal', '-surprisal', type=Path,
                        help='path to file containing token-based surprisals')
    parser.add_argument('--sentences', '-sentences', type=Path,
                        help='path to file containing sentences')
    parser.add_argument('--model', '-model', '--m', '-m', type=str,
                        choices=utils.MODELS, default=None, 
                        help='model for cpllab Docker container. '
                             'if None, then split tokens by spaces')
    parser.add_argument('--i', '-i', type=Path,
                        help='path to JSON file with input data')
    parser.add_argument('--o', '-o', type=Path,
                        help='path to JSON file to write output data')
    args = parser.parse_args()
    main(args)