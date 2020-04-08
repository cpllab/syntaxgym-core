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
        self.oovs = {region["region_number"]: [] for region in regions}

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
        drop_pattern = None
        if spec['tokenizer'].get('drop_token_pattern') is not None:
            drop_pattern = re.compile(spec['tokenizer']['drop_token_pattern'])

        # Sentinel: blindly add next N tokens to current region.
        skip_n = 0

        # iterate over all tokens in sentence
        t_idx = 0
        while t_idx < len(self.tokens):
            token = self.tokens[t_idx]

            if skip_n > 0:
                # Blindly add token to current region and continue.
                region2tokens[r.region_number].append(token)

                skip_n -= 1
                t_idx += 1
                continue

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
                t_idx += 1
                continue

            # skip current token for special cases
            elif token in spec['vocabulary']['special_types']:
                # TODO: which region should special_type associate with?
                t_idx += 1
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
            if drop_pattern is not None:
                content = re.sub(drop_pattern, '', content)

            # if empty region, proceed to next region (keeping current token)
            if content == '':
                r_idx += 1
                r, content = self.get_next_region(r_idx)

            # remove casing if necessary
            if not spec['tokenizer']['cased']:
                content = content.lower()

            # if token has exact match with beginning of content, or unk
            if content.startswith(token) or token in spec['vocabulary']['unk_types']:
                # add token to list of tokens for current region
                region2tokens[r.region_number].append(token)

                if content.startswith(token):
                    # remove token from content
                    content = content[len(token):]
                    t_idx += 1
                else:
                    # extract maximal string of OOVs by looking for match with
                    # next non-OOV token
                    tokens_remaining = len(self.tokens) - t_idx
                    oov_str = None
                    for token_window_size in range(1, tokens_remaining+1):
                        # token_window_size is number of tokens to look ahead
                        if token_window_size == tokens_remaining:
                            oov_str = content
                            skip_n = token_window_size
                            break
                        else:
                            if token_window_size > 1:
                                warnings.warn('Consecutive OOVs found in Item {}, Condition "{}"!'.format(
                                    self.item_num, self.condition_name
                                ), RuntimeWarning)

                            next_token = self.tokens[t_idx + token_window_size]

                            # Eat up content across regions until we come to a
                            # token that we can match with `next_token`.
                            eaten_content = []
                            for next_r_idx in range(r_idx, len(self.regions)):
                                if oov_str:
                                    break

                                if next_r_idx == r_idx:
                                    next_r_content = content
                                else:
                                    eaten_content.append(next_r_content.strip())
                                    _, next_r_content = self.get_next_region(next_r_idx)

                                for i in range(len(next_r_content)):
                                    if next_r_content[i:i+len(next_token)] == next_token:
                                        # We found a token which faithfully
                                        # matches the reference token. Break
                                        # just before that token.
                                        eaten_content.append(next_r_content[:i].strip())
                                        oov_str = " ".join(eaten_content).strip()

                                        # track OOVs -- put them in the
                                        # leftmost associated region
                                        self.oovs[r.region_number].extend(oov_str.split(" "))

                                        # Blindly add all these eaten tokens
                                        # from the content to the leftmost
                                        # region -- not including the token
                                        # that just matched, of course.
                                        region2tokens[r.region_number].extend(self.tokens[t_idx:t_idx + token_window_size])
                                        t_idx += token_window_size

                                        # Update the current region reference.
                                        r_idx = next_r_idx
                                        r = self.regions[r_idx]
                                        content = next_r_content[i:]

                                        break

                            if oov_str:
                                break


                    if content.strip() == '' and r_idx == len(self.regions) - 1:
                        return region2tokens

                # if end of content (removing spaces), and before last region
                if content.strip() == '' and r_idx < len(self.regions) - 1:
                    # warn user
                    if r_idx > 0 and self.oovs[r_idx]:
                        warnings.warn('OOVs found in Item {}, Condition "{}": "{}"'.format(
                            self.item_num, self.condition_name, self.oovs[r_idx]
                        ), RuntimeWarning)
                    r_idx += 1
                    r, content = self.get_next_region(r_idx)

            # otherwise, move to next region and token
            else:
                t_idx += 1
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