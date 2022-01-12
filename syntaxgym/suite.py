from __future__ import annotations

import json
from pprint import pformat
import warnings
from collections import defaultdict
import re
from typing import Dict, List, Optional, Iterator

import pandas as pd

from syntaxgym import utils
from syntaxgym.prediction import Prediction


class Suite(object):
    """
    A test suite represents a targeted syntactic evaluation experiment.

    For more information, see :ref:`architecture`.

    :ivar condition_names: A list of condition name strings
    :ivar region_names: An ordered list of region name strings
    :ivar items: An array of item ``dicts``, represented just as in a suite
        JSON representation. See :ref:`suite_json` for more information.
    :ivar predictions: A list of :class:`~syntaxgym.prediction.Prediction` objects.
    :ivar meta: A dict of metadata about this suite, represented just as in a
        suite JSON representation. See :ref:`suite_json` for more information.
    """

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
        predictions = [Prediction.from_dict(pred_i, i, suite_dict["meta"]["metric"])
                       for i, pred_i in enumerate(suite_dict["predictions"])]

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

    def as_dataframe(self, metric: str = None) -> pd.DataFrame:
        """
        Convert self to a data frame describing per-region surprisals.
        Only usable / sensible for Suite instances which have been evaluated
        with surprisals.

        Returns:
            A long Pandas DataFrame, one row per region, with columns:
                - item_number
                - condition_name
                - region_number
                - content
                - metric_value: per-region metric, specified in Suite meta or
                    overridden with `metric` argument
                - oovs: comma-separated list of OOV items
        """
        columns = ("item_number", "condition_name", "region_number", "content",
                   "metric_value", "oovs")
        index_columns = ["item_number", "condition_name", "region_number"]
        ret = []
        metric = metric or self.meta["metric"]

        for item in self.items:
            for condition in item["conditions"]:
                for region in condition["regions"]:
                    ret.append((
                        item["item_number"],
                        condition["condition_name"],
                        region["region_number"],
                        region["content"],
                        region["metric_value"][self.meta["metric"]],
                        ",".join(region["oovs"])
                    ))

        ret = pd.DataFrame(ret, columns=columns).set_index(index_columns)
        return ret

    def iter_sentences(self) -> Iterator[str]:
        """
        Iterate over all sentences in the suite in fixed order.
        """
        for item in self.items:
            for cond in item["conditions"]:
                regions = [region["content"].lstrip()
                           for region in cond["regions"]
                           if region["content"].strip() != ""]
                sentence = " ".join(regions)
                yield sentence

    def iter_region_edges(self) -> Iterator[List[int]]:
        """
        For each sentence in the suite, get list of indices of each region's
        left edge in the sentence.
        """
        for item in self.items:
            for cond in item["conditions"]:
                regions = [region["content"].lstrip()
                           for region in cond["regions"]]

                idx = 0
                ret = []
                for r_idx, region in enumerate(regions):
                    ret.append(idx)

                    region_size = len(region)
                    if region.strip() != "" and r_idx != 0:
                        # Add joining space
                        region_size += 1

                    idx += region_size

                yield ret

    def evaluate_predictions(self) -> Dict[int, Dict[Prediction, bool]]:
        """
        Compute prediction results for each item.

        Returns:
            results: a nested dict mapping ``(item_number => prediction =>
                prediction_result)``
        """

        result: Dict[int, Dict[Prediction, bool]] = {}
        for item in self.items:
            result[item["item_number"]] = {}
            for prediction in self.predictions:
                result[item["item_number"]][prediction] = prediction(item)

        return result

    def __eq__(self, other):
        return isinstance(other, Suite) and json.dumps(self.as_dict()) == json.dumps(other.as_dict())


MOSES_PUNCT_SPLIT_TOKEN = re.compile(r"^@([-,.])@$")
"""
Moses tokenizers split intra-token hyphens and decimal separators , and .
into separate tokens, using @ as a sentinel for detokenization. We account for
this when detokenizing here.
"""


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
        metaspace = spec["tokenizer"].get("metaspace")

        # Sentinel: blindly add next N tokens to current region.
        skip_n = 0

        # iterate over all tokens in sentence
        t_idx = 0
        while t_idx < len(self.tokens):
            token = self.tokens[t_idx]

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Token-level operations
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # append and continue upon encountering start-of-sentence token
            if token in spec['vocabulary']['prefix_types']:
                region2tokens[r.region_number].append(token)
                t_idx += 1
                continue

            # exit loop upon encountering end-of-sentence token
            elif token in spec['vocabulary']['suffix_types']:
                region2tokens[r.region_number].append(token)
                break

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
            if (content != '' and drop_pattern
                and re.sub(drop_pattern, '', content.split()[0]) == ''):
                content = ' '.join(content.split()[1:])

            # if empty region, proceed to next region (keeping current token)
            if content == '':
                r_idx += 1
                r, content = self.get_next_region(r_idx)
                continue

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

            if not token_match and metaspace is not None and token.startswith(metaspace):
                token_match = True

                # metaspace may end up as its own token or joined with
                # surrounding content. account for both cases.
                if len(token) > len(metaspace):
                    step_count = len(token) - len(metaspace)
                else:
                    # if metaspace was on its own, we don't need to advance the
                    # reference string -- the corresponding space was already
                    # stripped by lstrip() call above.
                    step_count = 0

            # Account for Moses sentinel if relevant.
            if "moses" in spec["tokenizer"].get("behaviors", []) \
                and MOSES_PUNCT_SPLIT_TOKEN.match(token):
                # Match. Step forward the number of characters between the Moses
                # @ sentinel.
                token_match = True
                stripped_token = MOSES_PUNCT_SPLIT_TOKEN.match(token).group(1)
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
                            # No fancy work needed here -- we're consuming the
                            # entire remainder of the string. Add to current
                            # region and quit.
                            region2tokens[r.region_number].extend(self.tokens[t_idx:])

                            oov_str = " ".join([content] + [r.content for r in self.regions[r_idx + 1:]])
                            self.oovs[r.region_number].extend(oov_str.split())

                            t_idx += token_window_size

                            break
                        else:
                            if token_window_size > 1:
                                warnings.warn('Consecutive OOVs found in Item {}, Condition "{}"!'.format(
                                    self.item_num, self.condition_name
                                ), RuntimeWarning)

                            next_token = self.tokens[t_idx + token_window_size]
                            next_token_is_punct = re.match(r"\W+", next_token)

                            # Eat up content across regions until we come to a
                            # token that we can match with `next_token`.
                            eaten_content = []
                            for next_r_idx in range(r_idx, len(self.regions)):
                                if oov_str:
                                    # OOV resolution is complete. Break.
                                    break

                                if next_r_idx == r_idx:
                                    next_r_content = content
                                else:
                                    eaten_content.append(next_r_content.strip())
                                    _, next_r_content = self.get_next_region(next_r_idx)

                                for i in range(len(next_r_content)):
                                    # When searching for a word-like token
                                    # (not punctuation), only allow matches at
                                    # token boundaries in region content.
                                    # This protects against the edge case where
                                    # a substring of the unk'ed token matches
                                    # a succeeding content in the token, e.g.
                                    #   content: "will remand and order"
                                    #   tokens: "will <unk> and order"
                                    #
                                    # See test case "remand test"
                                    if not next_token_is_punct and \
                                      (i > 0 and not re.match(r"\W", next_r_content[i - 1])):
                                        continue

                                    if next_r_content[i:i+len(next_token)] == next_token:
                                        # We found a token which faithfully
                                        # matches the reference token. Break
                                        # just before that token.
                                        eaten_content.append(next_r_content[:i].strip())
                                        # NB, we use `oov_str` as a sentinel
                                        # marking that the match is complete
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
                                # OOV resolution is complete. Break.
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
    multiple_space_re = re.compile(r"\s{2,}")

    def __init__(self, region_number=None, content='',
                 metric_value: Optional[Dict[str, float]] = None,
                 oovs: Optional[List[str]] = None):
        if self.boundary_space_re.search(content):
            raise ValueError("Region content has leading and/or trailing space."
                             " This is not allowed. Region content:  %r"
                             % (content,))
        elif self.multiple_space_re.search(content):
            raise ValueError("Region content has multiple consecutive spaces. "
                             "This is not allowed. Region content:  %r"
                             % (content,))

        utils.save_args(vars())
        self.token_surprisals = []

    def __repr__(self):
        s = 'Region(\n{}\n)'.format(pformat(vars(self)))
        return s

    def agg_surprisal(self, metric):
        self.surprisal = utils.METRICS[metric](self.token_surprisals)
        return self.surprisal
