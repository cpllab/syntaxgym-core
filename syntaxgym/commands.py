import json
import logging
import sys

import click

from lm_zoo import get_registry
import syntaxgym as S


def _prepare_model(model_ref, checkpoint=None):
    model = get_registry()[model_ref]
    if checkpoint is not None:
        return model.with_checkpoint(checkpoint)
    return model


class State(object):
    def __init__(self):
        self.verbose = False

pass_state = click.make_pass_decorator(State, ensure=True)


@click.group()
@click.option("--verbose", "-v", is_flag=True)
@pass_state
def syntaxgym(state, verbose):
    state.verbose = verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)


@syntaxgym.command(help="Compute per-region surprisals for a language model on the given test suite")
@click.argument("model")
@click.argument("suite_file", type=click.File("r"))
@click.option("--checkpoint")
@pass_state
def compute_surprisals(state, model, suite_file, checkpoint):
    model = _prepare_model(model, checkpoint)
    result = S.compute_surprisals(model, suite_file)
    json.dump(result.as_dict(), sys.stdout, indent=2)


@syntaxgym.command(help="Evaluate prediction results on the given test suite")
@click.argument("suite_file", type=click.File("r"))
@pass_state
def evaluate(state, suite_file):
    result = S.evaluate(suite_file)
    result.to_csv(sys.stdout, sep="\t")


@syntaxgym.command(help="Run the model and test suite through the full pipeline")
@click.argument("model")
@click.argument("suite_file", type=click.File("r"))
@click.option("--checkpoint")
@pass_state
def run(state, model, suite_file, checkpoint):
    model = _prepare_model(model, checkpoint)
    suite = S.compute_surprisals(model, suite_file)
    result = S.evaluate(suite)
    result.to_csv(sys.stdout, sep="\t")

