from copy import deepcopy
from io import StringIO
import itertools
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from syntaxgym import aggregate_surprisals, Suite
from syntaxgym.utils import TokenMismatch


surprisals_f = StringIO("""sentence_id\ttoken_id\ttoken\tsurprisal
1\t1            After   0.000000
1\t1              the   1.313512
1\t2              man   8.334731
1\t3              who   2.795280
1\t4                a   7.305301
1\t5           friend   3.945270
1\t6              had   2.615330
1\t7           helped   6.455917
1\t8             shot  10.508051
1\t9              the   2.094278
1\t10            bird   7.991671
1\t11            that   5.653132
1\t12              he   2.296663
1\t13             had   2.178579
1\t14            been   2.163770
1\t15        tracking   9.606250
1\t16        secretly  10.261303
1\t17               .   1.194046
1\t18           <eos>   0.007771""")
surprisals = pd.read_csv(surprisals_f, delim_whitespace=True)

suite = {
    "meta": {
        "name": "subordination_orc-orc",
        "metric": "sum",
        "comment": "",
        "reference": "",
        "author": "",
    },
    "region_meta": {
        "1": "Subordinate clause 1",
        "2": "subj_modifier",
        "3": "Subordinate clause 2",
        "4": "obj_modifier",
        "5": "Main clause"
    },
    "predictions": [
        {"type": "formula",
            "formula": "((5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ) & ((5;%sub_matrix%) < (5;%no-sub_matrix%))"}
    ],
    "items": [
        {
            "item_number": 1,
            "conditions": [
                {
                    "condition_name": "sub_no-matrix",
                    "regions": [
                        {
                            "region_number": 1,
                            "content": "After the man",
                            "metric_value": {
                                "sum": 18.504567167144707
                            }
                        },
                        {
                            "region_number": 2,
                            "content": "who a friend had helped",
                            "metric_value": {
                                "sum": 13.939815152698332
                            }
                        },
                        {
                            "region_number": 3,
                            "content": "shot the bird",
                            "metric_value": {
                                "sum": 11.948716490393831
                            }
                        },
                        {
                            "region_number": 4,
                            "content": "that he had been tracking secretly",
                            "metric_value": {
                                "sum": 11.217957010846014
                            }
                        },
                        {
                            "region_number": 5,
                            "content": ".",
                            "metric_value": {
                                "sum": 12.49306574242164
                            }
                        }
                    ]
                },
            ],
        },
    ],
}
suite = Suite.from_dict(suite)

tokens = ["After the man who a friend had helped shot the bird that he had been tracking secretly . <eos>".split()]
unks = [[0 for _ in tokens_i] for tokens_i in tokens]
spec = {
    "name": "lmzoo-basic-eos",
    "ref_url": "",

    "image": {
        "maintainer": "jon@gauthiers.net",
        "version": "NA",
        "datetime": "NA",
        "gpu": {
            "required": False,
            "supported": False
        }
    },

    "vocabulary": {
        "unk_types": ["<unk>"],
        "prefix_types": [""],
        "suffix_types": ["<eos>"],
        "special_types": [],
        "items": list(set((itertools.chain.from_iterable(tokens)))),
    },

    "tokenizer": {
        "type": "word",
        "cased": True
    }
}


def test_no_inplace():
    old_suite = deepcopy(suite)
    old_surprisals = surprisals.copy()

    result = aggregate_surprisals(surprisals, tokens, unks, suite, spec)

    assert suite == old_suite
    assert surprisals.equals(old_surprisals)


def test_basic():
    result = aggregate_surprisals(surprisals, tokens, unks, suite, spec)

    np.testing.assert_almost_equal(result.items[0]["conditions"][0]["regions"][0]["metric_value"]["sum"],
                                   surprisals.iloc[:3].surprisal.sum())


def test_tokenization_too_short():
    """
    throw error when tokens list missing tokens from surprisals list
    """
    # NB missing final <eos> token
    tokens = ["After the man who a friend had helped shot the bird that he had been tracking secretly .".split()]
    unks = [[0 for _ in tokens_i] for tokens_i in tokens]

    with pytest.raises(ValueError):
        aggregate_surprisals(surprisals, tokens, unks, suite, spec)


def test_tokenization_too_long():
    """
    throw error when surprisals list missing tokens from token list
    """
    tokens = ["After the man who a friend had helped shot the bird that he had been tracking secretly . <eos>".split()]
    unks = [[0 for _ in tokens_i] for tokens_i in tokens]

    surp = surprisals.copy()
    surp = surp[~(surp.token == "helped")]

    with pytest.raises(ValueError):
        aggregate_surprisals(surp, tokens, unks, suite, spec)


def test_mismatch():
    """
    throw error when regions and tokens can't be matched
    """
    tokens = ["After the man who a friend had hAAAlped shot the bird that he had been tracking secretly . <eos>".split()]
    unks = [[0 for _ in tokens_i] for tokens_i in tokens]

    surp = surprisals.copy()
    surp.loc[surp.token == "helped", "token"] = "hAAAlped"

    with pytest.raises(TokenMismatch):
        aggregate_surprisals(surp, tokens, unks, suite, spec)
