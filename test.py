import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import subprocess

from nose.tools import *


MODELS = ["grnn", "transformer-xl", "rnng", "jrnn"]
TEST_CASES = {
    "basic": (Path(__file__).parent / "data" / "test_in.json").open("r").read(),
}


def _get_sentences(input_json_f, sentences_f):
    return subprocess.run(f"python get_sentences.py --i {input_json_f.name} --o {sentences_f.name}".split(" "),
                          check=True)

def _get_surprisals(container, sentences_f, surprisals_f):
    return subprocess.run(
            f"docker run --rm -i {container} get_surprisals /dev/stdin".split(),
            check=True, stdin=sentences_f, stdout=surprisals_f)

def _agg_surprisals(model, input_json_f, sentences_f, surprisals_f, out_json_f):
    return subprocess.run(
            (f"python agg_surprisals.py --surprisal {surprisals_f.name} --sentences {sentences_f.name} "
             f"--model {model} --i {input_json_f.name} --o {out_json_f.name}").split(" "),
            check=True)


def _test_case(model_name, input_json):
    container = f"cpllab/language-models:{model_name}"

    with NamedTemporaryFile("w") as input_json_f, \
            NamedTemporaryFile("w+") as sentences_f, \
            NamedTemporaryFile("w+") as surprisals_f, \
            NamedTemporaryFile("w+") as out_json_f:

        input_json_f.write(input_json)
        input_json_f.flush()

        _get_sentences(input_json_f, sentences_f)
        _get_surprisals(container, sentences_f, surprisals_f)
        _agg_surprisals(model_name, input_json_f, sentences_f, surprisals_f, out_json_f)

        sentences = sentences_f.read()
        surprisals = surprisals_f.read()
        out_json = json.load(out_json_f)

    # TODO add tests :)
    ok_(out_json is not None)


def test_all_cases():
    for model in MODELS:
        for case_name, input_json in TEST_CASES.items():
            _test_case.description = f"{case_name}/{model}"
            yield _test_case, model, input_json

