import utils
import argparse
from pathlib import Path

def clean(surprisals, head):
    try:
        idx = surprisals.index(head)
        return surprisals[idx:]
    except:
        raise ValueError('string \"%s\" not found in file' % head)

def main(args):
    surprisals = utils.read_lines(args.i)
    surprisals = clean(surprisals, args.head)
    utils.write_lines(surprisals, args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean surprisal file')
    parser.add_argument('--i', '-i', type=Path,
                        help='path to input surprisal file')
    parser.add_argument('--o', '-o', type=Path,
                        help='path to write cleaned file')
    parser.add_argument('--head', '-head', type=str,
                        default='sentence_id\ttoken_id\ttoken\tsurprisal',
                        help='discard all lines before this string')
    args = parser.parse_args()
    main(args)