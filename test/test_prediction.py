import pytest

from copy import deepcopy

from syntaxgym.prediction import *

@pytest.fixture
def null_item():
    return {"conditions": [], "regions": []}


def test_prediction_parser(dummy_suite_json):
    p0 = Prediction(0, dummy_suite_json["predictions"][0]["formula"])
    p1 = Prediction(1, "(5;%sub_no-matrix%)>(5;%no-sub_no-matrix%)")

    item = dummy_suite_json["items"][0]
    old_item = deepcopy(item)

    assert not p0(item)
    assert old_item == item, "Prediction evaluation should not change item dict"
    assert not p0(item), "Prediction evaluation should yield consistent results"

    assert not p1(item)
    fudged_item = deepcopy(item)
    fudged_item["conditions"][0]["regions"][4]["metric_value"]["sum"] += 5000
    assert p1(fudged_item)


def test_parse_multiple_conjunction():
    p0 = Prediction(0, "((5;%reduced_ambig%) > (5;%unreduced_ambig%)) & ((5;%reduced_ambig%) > (5;%reduced_unambig%)) & (((5;%reduced_ambig%) - (5;%unreduced_ambig%)) > ((5;%reduced_unambig%) - (5;%unreduced_unambig%)))")
    assert str(p0) == "Prediction(((((5;%reduced_ambig%) > (5;%unreduced_ambig%)) & ((5;%reduced_ambig%) > (5;%reduced_unambig%))) & (((5;%reduced_ambig%) - (5;%unreduced_ambig%)) > ((5;%reduced_unambig%) - (5;%unreduced_unambig%)))))"
    # assert str(p0) == "Prediction((5;%reduced_ambig%) > (5;%unreduced_ambig%) & (5;%reduced_ambig%) > (5;%reduced_unambig%) & (5;%reduced_ambig%) - (5;%unreduced_ambig%) > (5;%reduced_unambig%) - (5;%unreduced_unambig%))"

def test_parse_multiple_subtraction(null_item):
    p0 = Prediction(0, "5 - 4 - 3 < 0")
    # NB this will evaluate to false if the third term is missed; true if it is not
    assert p0(null_item)

    p1 = Prediction(0, "5+4=9-1+1")
    assert p1(null_item)

    p2 = Prediction(0, "5-(3+2)=0")
    assert p2(null_item)

    p3 = Prediction(0, "(5-3)+2=4")
    assert p3(null_item)


def test_parens_maintained_by_serialization(null_item):
    """Converting a Prediction instance to a string and then re-parsing should yield the same result."""
    p4 = Prediction(0, "(5-3)+((2+1)-(2+2))=1")
    assert p4(null_item)

    p4_2 = Prediction(0, str(p4.formula))
    assert p4_2(null_item)


def test_prediction_parse_cleft():
    p0 = Prediction(0, "((7;%np_mismatch%)-(7;%np_match%))+(((6;%vp_mismatch%)+(7;%vp_mismatch%))-((6;%vp_match%)+(7;%vp_match%)))>0")
    p1 = Prediction(1, "((9;%what_nogap%) =  (9;%that_nogap%))& ((6;%what_subjgap%) =  (6;%that_subjgap%)) ")


def test_prediction_referenced_regions():
    p0 = Prediction(0, "((5;%reduced_ambig%) > (5;%unreduced_ambig%)) & ((5;%reduced_ambig%) > (5;%reduced_unambig%)) & (((5;%reduced_ambig%) - (5;%unreduced_ambig%)) > ((5;%reduced_unambig%) - (5;%unreduced_unambig%)))")
    referenced = p0.referenced_regions

    assert set(referenced) == {("reduced_ambig", 5), ("unreduced_ambig", 5), ("reduced_unambig", 5), ("unreduced_unambig", 5)}
    assert sorted(set(referenced)) == sorted(referenced), "No duplicates"
