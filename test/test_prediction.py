from copy import deepcopy

from syntaxgym.prediction import *


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


