from __future__ import annotations

import json
from pprint import pformat
import re
from typing import Dict, List, Optional, Iterator

import pandas as pd

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


class Sentence(object):
    def __init__(self, tokens, unks=None, item_num=None,
                 condition_name='', regions=None):
        self.tokens = tokens
        self.unks = unks
        self.item_num = item_num
        self.condition_name = condition_name
        self.regions = [Region(**r) for r in regions]
        self.content = ' '.join(r.content for r in self.regions)
        self.oovs = {region["region_number"]: [] for region in regions}

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((tuple(self.tokens),
                     tuple(self.unks) if self.unks is not None else None,
                     self.item_num,
                     self.condition_name,
                     tuple(self.regions),
                     self.content,
                     tuple((r_number, tuple(oovs))
                           for r_number, oovs in self.oovs.items())))

    def compute_region2tokens(self, spec: dict):
        """
        Infer the mapping between sentence tokens and regions based on
        heuristic algorithm.

        Used as backup when model tokenizers do not provide a detokenization
        procedure.
        """

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
        pass


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

        self.region_number: int = region_number
        self.content = content
        self.metric_value = metric_value
        self.oovs = oovs

    def __repr__(self):
        s = 'Region(\n{}\n)'.format(pformat(vars(self)))
        return s
