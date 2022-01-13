from copy import deepcopy
from io import StringIO
import itertools
from pprint import pprint

import pytest

import numpy as np
import pandas as pd
import lm_zoo as Z

from syntaxgym import aggregate_surprisals
from syntaxgym.agg_surprisals import compute_mapping_heuristic, compute_mapping_huggingface
from syntaxgym.suite import Suite, Region
from syntaxgym.utils import TokenMismatch, METRICS

from conftest import with_images


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

suite_json = {
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


@pytest.fixture
def suite():
    return Suite.from_dict(suite_json)


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


# TODO: aggregate-surprisals now requires `model` argument. need a dummy model
# shim here. lmzoo provides one but it's tedious with a fs setup :/
class DummyModel(Z.models.DummyModel):

    rets = {
        "tokenize": tokens,
        "get_surprisals": surprisals,
        "unkify": unks,
        "spec": spec,
    }

    def __init__(self, surprisals=None, tokens=None, unks=None):
        if surprisals is not None:
            self.rets["get_surprisals"] = surprisals
        if tokens is not None:
            self.rets["tokenize"] = tokens
        if unks is not None:
            self.rets["unkify"] = unks

    def get_result(self, command: str, *args):
        return self.rets[command]


model = DummyModel()


def test_no_inplace(suite):
    print(suite)
    old_suite = deepcopy(suite)
    old_surprisals = surprisals.copy()

    aggregate_surprisals(model, surprisals, tokens, suite)

    assert suite == old_suite
    assert surprisals.equals(old_surprisals)


@pytest.mark.parametrize("metric", ["sum", "mean"])
def test_basic(suite, metric):
    suite_ = deepcopy(suite)
    suite_.meta["metric"] = metric

    result = aggregate_surprisals(model, surprisals, tokens, suite_)

    metric_fn = METRICS[metric]
    np.testing.assert_almost_equal(result.items[0]["conditions"][0]["regions"][0]["metric_value"][metric],
                                   metric_fn(surprisals.iloc[:3].surprisal))


def test_tokenization_too_short(suite):
    """
    throw error when tokens list missing tokens from surprisals list
    """
    # NB missing final <eos> token
    tokens = ["After the man who a friend had helped shot the bird that he had been tracking secretly .".split()]
    model = DummyModel(tokens=tokens, unks=unks)

    with pytest.raises(ValueError):
        aggregate_surprisals(model, surprisals, tokens, suite)


def test_tokenization_too_long(suite):
    """
    throw error when surprisals list missing tokens from token list
    """
    tokens = ["After the man who a friend had helped shot the bird that he had been tracking secretly . <eos>".split()]

    surp = surprisals.copy()
    surp = surp[~(surp.token == "helped")]

    model = DummyModel(surprisals=surp, tokens=tokens)

    with pytest.raises(ValueError):
        aggregate_surprisals(model, surp, tokens, suite)


def test_mismatch(suite):
    """
    throw error when regions and tokens can't be matched
    """
    tokens = ["After the man who a friend had hAAAlped shot the bird that he had been tracking secretly . <eos>".split()]

    surp = surprisals.copy()
    surp.loc[surp.token == "helped", "token"] = "hAAAlped"

    with pytest.raises(TokenMismatch):
        aggregate_surprisals(model, surp, tokens, suite)



#############


@with_images("lmzoo-basic-eos-sos")
def test_mapping_eos_sos(client, built_model):
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "test."}
    ]
    regions = [Region(**r) for r in regions]
    spec = Z.spec(built_model)
    tokens = "<s> This is a test . </s>".split()

    mapping = compute_mapping_heuristic(tokens, regions, spec)
    assert mapping.region_to_tokens == {
        1: ["<s>", "This"],
        2: ["is"],
        3: ["a"],
        4: ["test", ".", "</s>"]
    }


@with_images("lmzoo-basic")
def test_unk(client, built_model):
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is WEIRDADVERB"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDNOUN."}
    ]
    regions = [Region(**r) for r in regions]
    spec = Z.spec(built_model)
    tokens = "This is <unk> a <unk> .".split()

    mapping = compute_mapping_heuristic(tokens, regions, spec)
    assert mapping.region_to_tokens == {
        1: ["This"],
        2: ["is", "<unk>"],
        3: ["a"],
        4: ["<unk>", "."],
    }

    assert mapping.oovs == {
        1: [],
        2: ["WEIRDADVERB"],
        3: [],
        4: ["WEIRDNOUN"],
    }


@with_images("lmzoo-basic")
def test_consecutive_unk(client, built_model):
    """
    Consecutive UNKs are mapped to regions by lookahead -- we look ahead in the
    token string for the next non-unk token, and associate all unks up to that
    token with the current region.
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDADVERB test WEIRDADJECTIVE WEIRDNOUN."}
    ]
    regions = [Region(**r) for r in regions]
    spec = Z.spec(built_model)
    tokens = "This is a <unk> test <unk> <unk> .".split()

    mapping = compute_mapping_heuristic(tokens, regions, spec)
    assert mapping.region_to_tokens == {
        1: ["This"],
        2: ["is"],
        3: ["a"],
        4: ["<unk>", "test", "<unk>", "<unk>", "."],
    }

    assert mapping.oovs == {
        1: [],
        2: [],
        3: [],
        4: ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"],
    }


@with_images("lmzoo-basic")
def test_consecutive_unk2(client, built_model):
    """
    consecutive unks in the middle of a region, with non-unks following in the
    same region
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDADVERB test WEIRDADJECTIVE WEIRDNOUN and some more"},
        {"region_number": 5, "content": "content."}
    ]
    regions = [Region(**r) for r in regions]
    spec = Z.spec(built_model)
    tokens = "This is a <unk> test <unk> <unk> and some more content .".split()

    mapping = compute_mapping_heuristic(tokens, regions, spec)
    assert mapping.region_to_tokens == {
        1: ["This"],
        2: ["is"],
        3: ["a"],
        4: ["<unk>", "test", "<unk>", "<unk>", "and", "some", "more"],
        5: ["content", "."],
    }

    assert mapping.oovs == {
        1: [],
        2: [],
        3: [],
        4: ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"],
        5: [],
    }


@with_images("lmzoo-basic")
def test_consecutive_unk3(client, built_model):
    """
    consecutive unks at the end of a region, with more regions after
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "WEIRDADVERB test WEIRDADJECTIVE WEIRDNOUN"},
        {"region_number": 5, "content": "and some more content."}
    ]
    regions = [Region(**r) for r in regions]
    spec = Z.spec(built_model)
    tokens = "This is a <unk> test <unk> <unk> and some more content .".split()

    mapping = compute_mapping_heuristic(tokens, regions, spec)
    assert mapping.region_to_tokens == {
        1: ["This"],
        2: ["is"],
        3: ["a"],
        4: ["<unk>", "test", "<unk>", "<unk>"],
        5: ["and", "some", "more", "content", "."],
    }

    assert mapping.oovs == {
        1: [],
        2: [],
        3: [],
        4: ["WEIRDADVERB", "WEIRDADJECTIVE", "WEIRDNOUN"],
        5: [],
    }


@with_images("lmzoo-basic")
def test_empty_regions2(client, built_model):
    """
    handle consecutive empty regions
    """
    regions = [
        {'content': 'Peter', 'region_number': 1},
        {'content': 'calls', 'region_number': 2},
        {'content': 'the candidates', 'region_number': 3},
        {'content': 'that', 'region_number': 4},
        {'content': 'the jury', 'region_number': 5},
        {'content': 'will firmly', 'region_number': 6},
        {'content': 'wait', 'region_number': 7},
        {'content': '', 'region_number': 8},
        {'content': '', 'region_number': 9},
        {'content': 'after the audition.', 'region_number': 10}
    ]
    regions = [Region(**r) for r in regions]
    spec = Z.spec(built_model)
    tokens = ['Peter', 'calls', 'the', 'candidates', 'that', 'the', 'jury',
              'will', 'firmly', 'wait', 'after', 'the', '<unk>']

    mapping = compute_mapping_heuristic(tokens, regions, spec)
    assert mapping.region_to_tokens == {
        1: ["Peter"],
        2: ["calls"],
        3: ["the", "candidates"],
        4: ["that"],
        5: ["the", "jury"],
        6: ["will", "firmly"],
        7: ["wait"],
        8: [],
        9: [],
        10: ["after", "the", "<unk>"],
    }

    assert mapping.oovs == {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: ["audition."],
    }


@with_images("lmzoo-basic-eos-sos-same")
def test_boundary_tokens_same(client, built_model):
    """
    Work with images whose boundary tokens are exactly the same at EOS and BOS.
    """
    regions = [
        {"region_number": 1, "content": "This"},
        {"region_number": 2, "content": "is"},
        {"region_number": 3, "content": "a"},
        {"region_number": 4, "content": "test."}
    ]
    regions = [Region(**r) for r in regions]
    spec = Z.spec(built_model)
    tokens = "<BOUNDARY> This is a test . <BOUNDARY>".split()

    mapping = compute_mapping_heuristic(tokens, regions, spec)
    assert mapping.region_to_tokens == {
        1: ["<BOUNDARY>", "This"],
        2: ["is"],
        3: ["a"],
        4 : ["test", ".", "<BOUNDARY>"]
    }


DYNAMIC_CASES = [

    ("Test empty regions",
     "lmzoo-basic",
     ["", "This", "is", "", "a test.", ""],
     None,
     {1: [], 2: ["This"], 3: ["is"], 4: [], 5: ["a", "test", "."], 6: []},
     {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}),

    ("Test consecutive empty regions",
     "lmzoo-basic",
     ["This", "is", "", "", "a test."],
     None,
     {1: ["This"], 2: ["is"], 3: [], 4: [], 5: ["a", "test", "."]},
     {1: [], 2: [], 3: [], 4: [], 5: []}),

    ("Test where end of unk matches start of next token (unk overlaps)",
     "lmzoo-basic",
     ["abc", "girlthat", "that", "xyz"],
     ["abc", "<unk>", "that", "xyz"],
     {1: ["abc"], 2: ["<unk>"], 3: ["that"], 4: ["xyz"]},
     {1: [], 2: ["girlthat"], 3: [], 4: []}),

    ("remand test (unk overlaps again)",
     "lmzoo-basic",
     ["will", "remand", "and", "order"],
     ["will", "<unk>", "and", "order"],
     {1: ["will"], 2: ["<unk>"], 3: ["and"], 4: ["order"]},
     {1: [], 2: ["remand"], 3: [], 4: []}),

    ("Test punctuation-only regions",
     "lmzoo-basic",
     "This is , a test .".split(" "),
     None,
     {1: ["This"], 2: ["is"], 3: [","], 4: ["a"], 5: ["test"], 6: ["."]},
     {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}),

    ("Test with uncased image",
     "lmzoo-basic-uncased",
     "This is a test.".split(" "),
     None,
     {1: ["this"], 2: ["is"], 3: ["a"], 4: ["test", "."]},
     {1: [], 2: [], 3: [], 4: []}),

    ("Test with punctuation-dropping image",
     "lmzoo-basic-nopunct",
     ["Mr. This", "is", ",", "a ---", "test", "."],
     None,
     {1: ["Mr.", "This"], 2: ["is"], 3: [], 4: ["a"], 5: ["test"], 6: []},
     {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}),

    ("Support BERT-style tokenization",
     "lmzoo-bert-tokenization",
     ["This is a test sentence."],
     None,
     {1: ["This", "is", "a", "test", "sen", "##tence", "."]},
     {1: []}),

    ("Support GPT2-style tokenization",
     "lmzoo-gpt-tokenization",
     ["This is a test sentence."],
     None,
     {1: ["This", "Ġis", "Ġa", "Ġtest", "Ġsen", "tence", "Ġ."]},
     {1: []}),

    ("Support GPT2-style tokenization across regions",
     "lmzoo-gpt-tokenization",
     ["This is a test sentence", "."],
     None,
     {1: ["This", "Ġis", "Ġa", "Ġtest", "Ġsen", "tence"],
      2: ["Ġ."]},
     {1: [], 2: []}),

    ("Support transfo-xl / Moses tokenization",
     "lmzoo-moses-tokenization",
     ["This is a test-ing", "sentence for $5,000.00."],
     None,
     {1: ["This", "is", "a", "test", "@-@", "ing"],
      2: ["sentence", "for", "$", "5,000.00", "."]},
     {1: [], 2: []}),

    ("Support transfo-xl / Moses tokenization with punct normalization",
     "lmzoo-moses-tokenization",
     ["This is a test–ing"],
     None,
     {1: ["This", "is", "a", "test", "@-@", "ing"]},
     {1: []}),

]


@pytest.mark.parametrize(argnames=("description", "image", "regions", "tokens",
                                   "expected_region2tokens", "expected_oovs"),
                         argvalues=DYNAMIC_CASES,
                         ids=[x[0] for x in DYNAMIC_CASES])
def test_dynamic_case(registry, client, description, image, regions, tokens,
                      expected_region2tokens, expected_oovs):
    if isinstance(image, str):
        image = image
        tag = "latest"
    else:
        image, tag = image
    model = registry[f"docker://{image}:{tag}"]

    # Preprocess regions list.
    if not isinstance(regions[0], dict):
        regions = [Region(region_number=i + 1, content=region)
                   for i, region in enumerate(regions)]

    if tokens is None:
        # Tokenize using image.
        tokens = Z.tokenize(model, [" ".join(r.content for r in regions)])[0]
        # with NamedTemporaryFile("w") as sentence_f:
        #     sentence_f.write(" ".join(r["content"] for r in regions) + "\n")
        #     sentence_f.flush()
        #
        #     tokens = tokenize_file(sentence_f.name, ":".join((image, tag)))[0]

    spec = Z.spec(model)

    print("Spec:")
    spec_to_print = deepcopy(spec)
    spec_to_print["vocabulary"]["items"] = ".... removed ...."
    pprint(spec_to_print)

    print("\n\nRegions:")
    pprint(regions)

    print("\n\nTokens:")
    print(tokens)

    mapping = compute_mapping_heuristic(tokens, regions, spec)

    if expected_region2tokens is not None:
        assert mapping.region_to_tokens == expected_region2tokens

    if expected_oovs is not None:
        assert mapping.oovs == expected_oovs


DYNAMIC_CASES_HUGGINGFACE = [

    ("reformer",
     "huggingface://hf-internal-testing/tiny-random-reformer",
     ["This is a testing", "sentence."],
     {1: ['▁', 'T', 'h', 'i', 's', '▁', 'i', 's', '▁a',
          '▁t', 'e', 's', 't', 'in', 'g'],
      2: ['▁', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', '.']},
     {1: [], 2: []}),

    ("pegasus",
     "huggingface://hf-internal-testing/tiny-random-pegasus",
     ["This is a testing", "sentence."],
     {1: ['▁', 'T', 'h', 'i', 's', '▁', 'i', 's', '▁', 'a',
          '▁', 't', 'e', 's', 't', 'ing'],
      2: ['▁', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', '.', '</s>']},
     {1: [], 2: []}),

]


def _make_suite_from_dynamic_case(regions):
    """
    Make a dummy single-item suite from the given test case's region list.
    """
    suite = Suite(
        condition_names=["a"],
        region_names=[str(r) for r in range(len(regions))],
        predictions=[],
        meta={"name": "dummy_suite"},
        items=[{"item_number": 1,
                "conditions": [
                    {"condition_name": "a",
                     "regions": regions}
                ]}]
    )
    return suite


@pytest.mark.parametrize(argnames=("description", "model_ref", "regions",
                                   "expected_region2tokens", "expected_oovs"),
                         argvalues=DYNAMIC_CASES_HUGGINGFACE,
                         ids=[x[0] for x in DYNAMIC_CASES_HUGGINGFACE])
def test_dynamic_case_huggingface(registry, description, model_ref,
                                  regions,
                                  expected_region2tokens, expected_oovs):
    """
    Run a dynamic test case using model lookup through LM Zoo (when no custom
    test image build is required.
    """
    # Preprocess regions list.
    if not isinstance(regions[0], dict):
        regions = [{"region_number": i + 1, "content": region}
                   for i, region in enumerate(regions)]

    model: Z.models.HuggingFaceModel = registry[model_ref]

    suite = _make_suite_from_dynamic_case(regions)
    spec = Z.spec(model)

    regions = [Region(**r) for r in regions]

    print("Spec:")
    spec_to_print = deepcopy(spec)
    spec_to_print["vocabulary"]["items"] = ".... removed ...."
    pprint(spec_to_print)

    print("\n\nRegions:")
    pprint(regions)

    sentence = list(suite.iter_sentences())[0]
    print(sentence)
    region_edges = list(suite.iter_region_edges())[0]
    tokenized = model.tokenizer.encode_plus(sentence,
                                            add_special_tokens=True,
                                            return_offsets_mapping=True)

    print("\n\nTokens:")
    print(tokenized.tokens())

    mapping = compute_mapping_huggingface(
        tokenized.tokens(), regions,
        token_offsets=tokenized["offset_mapping"],
        region_edges=region_edges)

    if expected_region2tokens is not None:
        assert mapping.region_to_tokens == expected_region2tokens

    if expected_oovs is not None:
        assert mapping.oovs == expected_oovs
