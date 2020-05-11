import json
import tempfile

import pytest

from utils import *


@pytest.fixture(scope="function")
def sentence_file():
    with tempfile.NamedTemporaryFile("w") as f:
        f.write("This is a test sentence.\nThis is a second test sentence.\n")
        f.flush()

        yield f


@pytest.fixture()
def image():
    return "cpllab/language-models:grnn"


def test_get_spec(image):
    get_spec(image)


def test_tokenize(sentence_file, image):
    expected = "This is a test sentence . <eos>\nThis is a second test sentence . <eos>"
    expected = expected.split("\n")
    expected = [sentence.split(" ") for sentence in expected]

    assert tokenize_file(sentence_file.name, image) == expected
