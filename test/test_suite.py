from copy import deepcopy
from functools import lru_cache
from io import StringIO
import json
from pathlib import Path
from pprint import pprint
import sys
from tempfile import NamedTemporaryFile

import jsonschema
import pandas as pd
import pytest
import requests

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)
L = logging.getLogger(__name__)

import lm_zoo as Z

from syntaxgym.suite import Suite, Sentence, Region

from conftest import LM_ZOO_IMAGES, with_images


SPEC_SCHEMA_URL = "https://cpllab.github.io/lm-zoo/schemas/language_model_spec.json"

DUMMY_SPEC_PATH = Path(__file__).parent / "dummy_specs"



##################################


@pytest.fixture(scope="module")
def spec_schema():
    return requests.get(SPEC_SCHEMA_URL).json()


@with_images(*LM_ZOO_IMAGES)
def test_specs(client, registry, built_image, spec_schema):
    """
    Validate specs against the lm-zoo standard.
    """
    jsonschema.validate(instance=Z.spec(registry[f"docker://{built_image}"]),
                        schema=spec_schema)


@pytest.mark.parametrize("region_str", ["test ", " test", " "])
def test_region_spaces(region_str):
    """
    Regions with leading/trailing spaces should be rejected
    """
    with pytest.raises(ValueError):
        Region(content=region_str)


# def test_special_types():
#     # TODO: if at region boundary, which region do we associate them with?
#     pass


def test_suite_as_dataframe(dummy_suite_json, dummy_suite_csv):
    suite = Suite.from_dict(dummy_suite_json)
    df = suite.as_dataframe()

    expected_df = pd.read_csv(StringIO(dummy_suite_csv), keep_default_na=False) \
        .set_index(df.index.names)

    pd.testing.assert_frame_equal(df, expected_df)
