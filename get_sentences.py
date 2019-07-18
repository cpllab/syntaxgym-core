import utils
import argparse
from pathlib import Path

def get_sentences(in_data):
    sentences = []
    for item in in_data['items']:
        for cond in item['conditions']:
            regions = [region['content'] for region in cond['regions']]
            sentence = ' '.join(regions)
            sentences.append(sentence)
    return sentences

def main(args):
    in_data = utils.load_json(args.i)
    sentences = get_sentences(in_data)
    utils.write_lines(sentences, args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make sentences from JSON file')
    parser.add_argument('--i', '-i', type=Path,
                        help='path to JSON file with input data')
    parser.add_argument('--o', '-o', type=Path,
                        help='path to file to write output data')
    args = parser.parse_args()
    main(args)