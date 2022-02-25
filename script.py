#!/usr/bin/env python

#%%
import navis
import numpy as np
import pandas as pd
import pymaid
from matplotlib import pyplot as plt

import os
import pickle
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Iterable, List, Tuple, TypeVar
from doctest import NORMALIZE_WHITESPACE
from black import out
from copy import deepcopy
from ipywidgets import IntProgress
import json



logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "output"
#logging tracker of events/errors; returns parent directory of given path, provides cache and output directories

SIDE_PLACEHOLDER = "__SIDE__"
#placeholder for merging of paired neurons

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
#loads stored catmaid credentials


### LR connectivity analysis ###



## Define functions


#%%
def get_neurons():
    # get CatmaidNeuronLists of left and right brain pairs, N.B. each is in arbitrary order
    # sw annotation has 1172 pairs
    
    fpath = CACHE_DIR / "Cneurons.pickle"
    if not os.path.exists(fpath):
        import pymaid
        neurons = tuple(pymaid.get_neuron("annotation:sw;brainpair;" + side) for side in "LR")

        with open(fpath, "wb") as f:
            pickle.dump(neurons, f, protocol=5)
    else:
        with open(fpath, "rb") as f:
            neurons = pickle.load(f)

    return neurons

def side_merge(left_neurons, right_neurons):
    # replace side name with placeholder, merge pairs and sort tuples
    '''TODO: using this SIDE_PLACEHOLDER misses ~500 pairs (not all have same names, nor use left/right as side differential)
    Instead wish to use above annotation (sw;brainpair;L or R) to merge pairs, should get all 1172'''

    unsided_l = {n.name.replace("left", SIDE_PLACEHOLDER): n for n in left_neurons if "left" in n.name}
    unsided_r = {n.name.replace("right", SIDE_PLACEHOLDER): n for n in right_neurons if "right" in n.name}
    in_both = set(unsided_l).intersection(unsided_r)
    for unsided_name in sorted(in_both):
        yield unsided_name, (unsided_l[unsided_name], unsided_r[unsided_name])


def generate_adj_matrix():
    # generate and load adjacency matrix for each pair (assum
    # TODO: make sure two lists are same length, and rows match (except for L/R difference)

    neurons_l, neurons_r = get_neurons()
    paired = dict(side_merge(neurons_l, neurons_r))

    all_neurons = []
    for left_right in paired.values():
        all_neurons.extend(left_right)

    adj = pymaid.adjacency_matrix(all_neurons, fractions=True, use_connectors=True)  # maybe use fractions=True
    key = [n.name for n in all_neurons]
    adj.index = key
    adj.columns = key
    return paired, adj

def generate_similarity(paired=None, adj=None, metric="cosine", is_input=False):
    out_file = OUT_DIR / f"sim_{metric}_{'in' if is_input else 'out'}put.json"
    if out_file.is_file():
        with open(out_file) as f:
            return json.load(f)

    if paired is None or adj is None:
        paired, adj = generate_adj_matrix()

    if is_input:
        adj = adj.T

    left_first = list(adj.index)
    right_first = []
    for left_idx in range(0, len(left_first), 2):
        right_first.append(left_first[left_idx + 1])
        right_first.append(left_first[left_idx])

    out = dict()

    for unsided_name, (left_nrn, right_nrn) in paired.items():
        left_array = np.asarray(adj.loc[left_nrn.name])
        right_array = np.asarray(adj.loc[right_nrn.name][right_first])
        sim = navis.connectivity_similarity(np.array([left_array, right_array]), metric="cosine")
        sim_arr = sim.to_numpy()
        val = float(sim_arr[0, 1])
        if np.isnan(val):
            val = None

        out[unsided_name] = val

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    return out


METRIC = "cosine"

paired, adj = generate_adj_matrix()

output_sim = generate_similarity(paired, adj, METRIC, is_input=False)
input_sim = generate_similarity(paired, adj, METRIC, is_input=True)


def sim_to_xy(sim, normalise=True):
    x = sorted(v for v in sim.values() if v is not None and not np.isnan(v))
    y = np.arange(len(x))
    if normalise:
        y /= len(x)
    return x, y


from matplotlib import pyplot as plt

NORMALISE = False

fig = plt.figure()
ax = fig.add_subplot()


out_x, out_y = sim_to_xy(output_sim, NORMALISE)
ax.plot(out_x, out_y, label="output similarity")

in_x, in_y = sim_to_xy(input_sim, NORMALISE)
ax.plot(in_x, in_y, label="input similarity")

ax.legend()
ax.set_xlabel(f"{METRIC} similarity value")
ax.set_label("Cumulative frequency")

plt.show()

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    return out


## Run analysis and explore data ##

#%%
METRIC = "cosine"

paired, adj = generate_adj_matrix()

output_sim = generate_similarity(paired, adj, METRIC, is_input=False)
input_sim = generate_similarity(paired, adj, METRIC, is_input=True)


def sim_to_xy(sim, normalise=True):
    x = sorted(v for v in sim.values() if not np.isnan(v))
    y = np.arange(len(x))
    if normalise:
        y /= len(x)
    return x, y



NORMALISE = False

fig = plt.figure()
ax = fig.add_subplot()


out_x, out_y = sim_to_xy(output_sim, NORMALISE)
ax.plot(out_x, out_y, label="output similarity")

in_x, in_y = sim_to_xy(input_sim, NORMALISE)
ax.plot(in_x, in_y, label="input similarity")

ax.legend()
ax.set_xlabel(f"{METRIC} similarity value")
ax.set_label("Cumulative frequency")

plt.show()