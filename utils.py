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

def write_lines(lines, path):
    """
    Writes list to Path, separated by \n.
    """
    with path.open('w') as f:
        for l in lines:
            f.write(str(l) + '\n')