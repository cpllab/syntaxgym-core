import json
import argparse
from pathlib import Path

def get_sentences(in_data):
    sentences = []
    for item in in_data.items:
        for cond in item['conditions']:
            regions = [region['content'].lstrip() for region in cond['regions'] if region['content'].strip() != '']
            sentence = ' '.join(regions)
            sentences.append(sentence)
    return sentences

def main(args):
    with open(args.i, 'r') as f:
        in_data = json.load(f)
    sentences = get_sentences(in_data)
    with open(args.o, 'w') as f:
        for s in sentences:
            f.write(s+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make sentences from JSON file')
    parser.add_argument('--i', '-i', type=str,
                        help='path to JSON file with input data')
    parser.add_argument('--o', '-o', type=str,
                        help='path to file to write output data')
    args = parser.parse_args()
    main(args)
