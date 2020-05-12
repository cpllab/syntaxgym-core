import json
import tempfile

import pytest

from syntaxgym import utils


@pytest.fixture(scope="function")
def sentence_file():
    with tempfile.NamedTemporaryFile("w") as f:
        f.write("This is a test sentence.\nThis is a second test sentence.\n")
        f.flush()

        yield f


@pytest.fixture(scope="module", params=["cpllab/language-models:grnn"])
def image(request):
    image_ref, tag = request.param.rsplit(":", 1)
    utils._pull_container(image_ref, tag)
    return ":".join((image_ref, tag))


def test_get_spec(image):
    utils.get_spec(image)


def test_tokenize(sentence_file, image):
    expected = "This is a test sentence . <eos>\nThis is a second test sentence . <eos>"
    expected = expected.split("\n")
    expected = [sentence.split(" ") for sentence in expected]

    assert utils.tokenize_file(sentence_file.name, image) == expected
