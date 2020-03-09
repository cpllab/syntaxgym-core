import utils
from pprint import pformat
import warnings
from collections import defaultdict
# import re

# punct_re = re.compile(r"^[-.?!,\*]+$")

class Sentence:
    def __init__(self, spec, tokens, unks, item_num=None, condition_name='', regions=None):
        self.tokens = tokens
        self.unks = unks
        self.item_num = item_num
        self.condition_name = condition_name
        self.regions = [Region(**r) for r in regions]
        self.content = ' '.join(r.content for r in self.regions)
        self.oovs = defaultdict(list)

        # compute region-to-token mapping upon initialization
        self.region2tokens = self.tokenize_regions(spec) 
        for i, r in enumerate(self.regions):
            r.tokens = self.region2tokens[r.region_number]
            self.regions[i] = r

    def tokenize_regions(self, spec):
        """
        Converts self.tokens (list of tokens) to dictionary of
        <region_number, token list> pairs.
        """
        # initialize counter, current region, and content
        r_idx = 0
        r = self.regions[r_idx]
        content = r.content

        # initialize region-to-token dict
        region_tokens = {r.region_number : [] for r in self.regions}

        # iterate over all tokens in sentence
        for t_idx, token in enumerate(self.tokens):
            # exit loop upon encountering end-of-sentence token
            if token in spec['vocabulary']['suffix_types']:
                region_tokens[r.region_number].append(token)
                break

            # proceed to next token for special cases
            elif token in spec['vocabulary']['prefix_types'] or token in spec['vocabulary']['special_types']:
                continue

            # warn user if token is out-of-vocabulary
            elif self.unks[t_idx]:
                next_word = content.split()[0]
                warnings.warn('OOV found in Item {}, Condition "{}": "{}"'.format(
                    self.item_num, self.condition_name, next_word
                ), RuntimeWarning)
                self.oovs[r_idx].append(next_word)

            # HACK: remove punctuation
            # if nopunct:
            #     content = ' '.join([s for s in content.split(' ') if not punct_re.match(s)])

            # HACK: quick, untested, dirty hack for BPE tokenization, e.g.
            # This token will decompose. --> ĠThis Ġtoken Ġwill Ġdecom pose .
            # if bpe and token.startswith('Ġ'):
            #     # remove token boundary
            #     token = token[1:]

            # remove leading spaces of current content
            content = content.lstrip()

            # if empty region, proceed to next region (keeping current token)
            if content == '':
                r_idx += 1
                r = self.regions[r_idx]
                content = r.content.lstrip()

            # remove casing if necessary
            if spec['cased']:
                content = content.lower()
            
            # if token has exact match with beginning of content, or unk
            if content.startswith(token) or token in spec['vocabulary']['unk_types']:
                # add token to list of tokens for current region
                region_tokens[r.region_number].append(token)
                
                # remove token from content
                if content.startswith(token):
                    content = content[len(token):]
                else:
                    # for unk token, skip the first token of content
                    content = ' '.join(content.split()[1:])

                # if end of content (removing spaces), and before last region
                if content.strip() == '' and r_idx < len(self.regions) - 1:
                    r_idx += 1
                    r = self.regions[r_idx]
                    content = r.content
                    
            # otherwise, move to next region and token
            else:
                r_idx += 1
                r = self.regions[r_idx]
                content = r.content
        return region_tokens

class Region:
    def __init__(self, region_number=None, content='', metric_value=None):
        utils.save_args(vars())
        self.token_surprisals = []

    def __repr__(self):
        s = 'Region(\n{}\n)'.format(pformat(vars(self)))
        return s

    def agg_surprisal(self, metric):
        self.surprisal = utils.METRICS[metric](self.token_surprisals)
        return self.surprisal