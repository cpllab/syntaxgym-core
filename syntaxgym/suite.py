from __future__ import annotations

import json
from pprint import pformat
import warnings
from collections import defaultdict
import re
from typing import Dict

from syntaxgym import utils
from syntaxgym.prediction import Prediction


class Suite(object):

    def __init__(self, condition_names, region_names, items, predictions, meta):
        self.condition_names = condition_names
        self.region_names = region_names
        self.items = items
        self.predictions = predictions
        self.meta = meta

    @classmethod
    def from_dict(cls, suite_dict):
        condition_names = [c["condition_name"] for c in suite_dict["items"][0]["conditions"]]
        region_names = [name for number, name
                        in sorted([(int(number), name)
                                   for number, name in suite_dict["region_meta"].items()])]
        items = suite_dict["items"]
        predictions = [Prediction.from_dict(pred_i, i) for i, pred_i in enumerate(suite_dict["predictions"])]

        return cls(condition_names=condition_names,
                   region_names=region_names,
                   items=items,
                   predictions=predictions,
                   meta=suite_dict["meta"])

    def as_dict(self):
        ret = dict(
            meta=self.meta,
            region_meta={i + 1: r for i, r in enumerate(self.region_names)},
            predictions=[p.as_dict() for p in self.predictions],
            items=self.items,
        )

        return ret

    def evaluate_predictions(self) -> Dict[int, Dict[Prediction, bool]]:
        """
        Compute prediction results for each item.

        Returns:
            results: a nested dict mapping ``(item_number => prediction =>
                prediction_result)``
        """

        result = {}
        for item in self.items:
            result[item["item_number"]] = {}
            for prediction in self.predictions:
                result[item["item_number"]][prediction] = prediction(item)

        return result

    def __eq__(self, other):
        return isinstance(other, Suite) and json.dumps(self.as_dict()) == json.dumps(other.as_dict())


class Sentence(object):
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
        if spec['tokenizer'].get('drop_token_pattern') is not None:
            drop_pattern = re.compile(spec['tokenizer']['drop_token_pattern'])
        else:
            drop_pattern = None

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

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Content-level operations
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # remove leading spaces of current content
            content = content.lstrip()

            # drop characters specified by regex (content up to next space)
            if drop_pattern and re.sub(drop_pattern, '', content.split()[0]) == '':
                content = ' '.join(content.split()[1:])

            # if empty region, proceed to next region (keeping current token)
            if content == '':
                r_idx += 1
                r, content = self.get_next_region(r_idx)

            # remove casing if necessary
            if not spec['tokenizer']['cased']:
                content = content.lower()

            # Check for a token match at the left edge of the region.
            step_count = None
            token_match = content.startswith(token)
            if token_match:
                # Exact match. We'll walk forward this number of characters
                step_count = len(token)
            # Subword tokenizers may have initial / final content that blocks
            # the match. Check again.
            if not token_match and spec["tokenizer"]["type"] == "subword":
                if spec["tokenizer"]["sentinel_position"] in ["initial", "medial"]:
                    stripped_token = token.lstrip(spec["tokenizer"]["sentinel_pattern"])
                elif spec["tokenizer"]["sentinel_position"] == "final":
                    stripped_token = token.rstrip(spec["tokenizer"]["sentinel_pattern"])

                token_match = content.startswith(stripped_token)
                # Soft subword match. Step forward the number of characters in
                # the matched subword, correcting for sentinel
                if token_match:
                    step_count = len(stripped_token)

            # If we found a left-edge match, or this is an unk
            if token_match or token in spec['vocabulary']['unk_types']:

                # First: consume the (soft) matched token.

                if token_match:
                    # add token to list of tokens for current region
                    region2tokens[r.region_number].append(token)

                    # remove token from content
                    content = content[step_count:]
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

class Region(object):
    boundary_space_re = re.compile(r"^\s|\s$")

    def __init__(self, region_number=None, content='', metric_value=None):
        if self.boundary_space_re.search(content):
            raise ValueError("Region content has leading and/or trailing space."
                             " This is not allowed. Region content:  %r"
                             % (content,))

        utils.save_args(vars())
        self.token_surprisals = []

    def __repr__(self):
        s = 'Region(\n{}\n)'.format(pformat(vars(self)))
        return s

    def agg_surprisal(self, metric):
        self.surprisal = utils.METRICS[metric](self.token_surprisals)
        return self.surprisal
