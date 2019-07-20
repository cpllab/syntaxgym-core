import json
import numpy as np
from inspect import getargspec

METRICS = {
    'sum': sum,
    'mean': np.mean,
    'median': np.median,
    'range': np.ptp,
    'max': max,
    'min': min
}

MODELS = ['grnn', 'transformer-xl', 'rnng', 'jrnn']

def flatten(l):
    """
    Flattens a list. Credit to https://stackoverflow.com/a/952952
    """
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

def save_args(values):
    """
    Automatically saves constructor arguments to object.
    Credit to https://stackoverflow.com/a/15484172
    """
    for i in getargspec(values['self'].__init__).args[1:]:
        setattr(values['self'], i, values[i])

def load_json(path):
    """
    Loads Path to JSON file as dictionary.
    """
    with path.open() as f:
        d = json.load(f)
    return d

def write_json(d, path):
    """
    Writes dictionary to JSON file specified by Path.
    """
    with path.open('w') as f:
        json.dump(d, f, indent=4)

def read_lines(path):
    """
    Reads lines from file into list with leading and trailing whitespace removed.
    """
    with path.open() as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

def write_lines(lines, path):
    """
    Writes list to Path, separated by \n.
    """
    with path.open('w') as f:
        for l in lines:
            f.write(str(l) + '\n')

def validate_metrics(metrics):
    # if only one metric specified, convert to list
    metrics = [metrics] if len(metrics) == 1 else metrics
    if any(m not in METRICS for m in metrics):
        bad_metrics = [m for m in metrics if m not in METRICS]
        raise ValueError('Unknown metrics: {}'.format(bad_metrics))