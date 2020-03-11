import utils
from pprint import pformat
import warnings
from collections import defaultdict
import re

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

    def get_next_region(self, r_idx):
        """
        Basic helper function for tokenize_regions.
        """
        r = self.regions[r_idx]
        return r, r.content

    def tokenize_regions(self, spec):
        """
        Converts self.tokens (list of tokens) to dictionary of
        <region_number, token list> pairs.
        """
        # initialize variables
        r_idx = 0
        r, content = self.get_next_region(r_idx)
        region2tokens = defaultdict(list)

        # compile regex for dropping
        drop_pattern = re.compile(spec['vocabulary']['drop_token_pattern'])

        # iterate over all tokens in sentence
        for t_idx, token in enumerate(self.tokens):

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Token-level operations
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # exit loop upon encountering end-of-sentence token
            if token in spec['vocabulary']['suffix_types']:
                region2tokens[r.region_number].append(token)
                break

            # append and continue upon encountering start-of-sentence token
            elif token in spec['vocabulary']['prefix_types']:
                region2tokens[r.region_number].append(token)
                continue
            
            # skip current token for special cases
            elif token in spec['vocabulary']['special_types']:
                # TODO: which region should special_type associate with?
                continue

            # HACK: quick, untested, dirty hack for BPE tokenization, e.g.
            # This token will decompose. --> ĠThis Ġtoken Ġwill Ġdecom pose .
            # if spec['tokenizer']['type'] == 'subword':
            #     if spec['tokenizer']['sentinel_position'] == 'initial' and token.startswith(spec['tokenizer']['sentinel_pattern']):
            #         # remove token boundary
            #         token = token[len(spec['tokenizer']['sentinel_pattern']):]
            #     elif spec['tokenizer']['sentinel_position'] == 'medial' and token.startswith(spec['tokenizer']['sentinel_pattern']):

            #             "tokenizer": {
            # "type": "subword",
            # "sentinel_pattern": "##",
            # "sentinel_position": "initial" || "medial" || "final",
            # }
            # if bpe and token.startswith('Ġ'):
            #     # remove token boundary
            #     token = token[1:]

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Content-level operations
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # remove leading spaces of current content
            content = content.lstrip()

            # drop characters specified by regex
            content = re.sub(drop_pattern, '', content)

            # if empty region, proceed to next region (keeping current token)
            if content == '':
                r_idx += 1
                r, content = self.get_next_region(r_idx)

            # remove casing if necessary
            if not spec['cased']:
                content = content.lower()
            
            # if token has exact match with beginning of content, or unk
            if content.startswith(token) or token in spec['vocabulary']['unk_types']:
                # add token to list of tokens for current region
                region2tokens[r.region_number].append(token)
                
                if content.startswith(token):
                    # remove token from content
                    content = content[len(token):]
                else:
                    # find OOVs by looking for match with next non-OOV token
                    tokens_remaining = len(self.tokens) - t_idx
                    oov_str = None
                    for token_window_size in range(1, tokens_remaining+1):
                        # token_window_size is number of tokens to look ahead
                        if token_window_size == tokens_remaining:
                            oov_str = content
                            break
                        else:
                            if token_window_size > 1:
                                warnings.warn('Consecutive OOVs found in Item {}, Condition "{}"!'.format(
                                    self.item_num, self.condition_name
                                ), RuntimeWarning)
                            for i in range(len(content)):
                                next_token = self.tokens[t_idx + token_window_size]
                                if content[i:i+len(next_token)] == next_token:
                                    oov_str = content[:i].strip()
                                    break
                            if oov_str:
                                break
                    # add OOVs to self.oovs
                    self.oovs[r_idx].append(oov_str)

                    # remove OOVs from content
                    content = content[len(oov_str):]

                    if content.strip() == '' and r_idx == len(self.regions) - 1:
                        return region2tokens

                # if end of content (removing spaces), and before last region
                if content.strip() == '' and r_idx < len(self.regions) - 1:
                    # warn user
                    if self.oovs[r_idx]:
                        warnings.warn('OOVs found in Item {}, Condition "{}": "{}"'.format(
                            self.item_num, self.condition_name, self.oovs[r_idx]
                        ), RuntimeWarning)
                    r_idx += 1
                    r, content = self.get_next_region(r_idx)
                    
            # otherwise, move to next region and token
            else:
                r_idx += 1
                r, content = self.get_next_region(r_idx)

        return region2tokens

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