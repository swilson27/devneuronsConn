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

def indexed_names(skids):
    # Taking skids, outputs a chronologically indexed order of skids:names as a dict
    names = [pymaid.get_names(skid) for skid in skids]
    dict = {k:v for x in names for k,v in x.items()}
    return(dict)

def side_merge(left_neurons, right_neurons):
    # takes left and right neurons (as respective skid:name dicts), and replaces side names with longest common substring
    # sort tuples

    left_names = indexed_names(left_neurons)
    right_names = indexed_names(right_neurons)
    name_pairs = list(zip(left_names.values(), right_names.values()))

    from difflib import SequenceMatcher
    merged_l=[]
    merged_r=[]
    for index, pair in enumerate(name_pairs):
        matcher = SequenceMatcher(None, pair[0], pair[1])
        merge = matcher.find_longest_match(0, len(pair[0]), 0, len(pair[1]))
        merged_l.append(pair[0][merge.a:merge.a + merge.size])
        #merged_r.append(
    in_both = set(merged_l).intersection(merged_r)
    for unsided_name in sorted(in_both): # yields unsided name, left name, right name for these intersected pairs
        yield unsided_name, (merged_l[unsided_name], merged_r[unsided_name])

def generate_adj_matrix(neurons_l, neurons_r):
    # generate and load adjacency matrix for each pair
    # TODO: assumes two lists are same length and rows match (except for L/R difference)

    #neurons_l, neurons_r = get_neurons()
    paired = dict(side_merge(neurons_l, neurons_r))

    all_neurons = []
    for left_right in paired.values():
        all_neurons.extend(left_right)

    adj = pymaid.adjacency_matrix(all_neurons, fractions=True, use_connectors=True)  # maybe use fractions=True
    key = [n.name for n in all_neurons]
    adj.index = key
    adj.columns = key
    return paired, adj
#%%

def merged_analysis(pair_list):
    # takes as input a list of tuples, assumed to be skeleton IDs
    # merges pairs, constructs adjacency matrix, scores connectivity (cosine)
    pair_dict = {}
    for pair in pair_list:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    
    cos_sims = {}
    for pair in pair_list:
        partners = pymaid.get_partners(pair[0:1])
        partners.append = pymaid.get_partners(pair[1])    
        partner_skids = set(partners["skeleton_id"])
        columns = []
        while len(partner_skids) > 0:
            skidcycle = iter(partner_skids)
            skid1 = next(skidcycle)
            skid2 = pair_dict.get(skid1, None)
            if skid2 is None:
                # Nothing to fuse. But must normalize
                columns.append(partners["skid1"] / float(sum(partners["skid1"])))
            else:
                # Fuse two vectors, normalized
                n_syn1 = float(sum(partners["skid1"]))
                n_syn2 = float(sum(partners["skid2"]))
                columns.append([v1/n_syn1 + v2/n_syn2 for v1, v2 in zip(partners["skid1"], partners["skid2"])])
            # Remove both from set
            del partner_skids[skid1]
            del partner_skids[skid2]
        # Two rows, as many colums as were needed
        matrix = np.array(columns).transpose() # rows have to be skid1, skid2

        score = navis.connectivity_similarity(matrix, metric="cosine")
        cos_sims[skid1] = score[0]
        cos_sims[skid2] = score[1]

    # Save the similarity scores
    # TODO create panda DataFrame with int column for skids and float column for scores, and save as CSV
            
        out = pd.DataFrame(columns = ["skid_left", "skid_right", "score"])
        
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


## Run analysis and explore data ##


#%%
METRIC = "cosine"

bp = pd.read_csv (r"/Users/swilson/Documents/Devneurons/MAnalysis/brain-pairs.csv")
bp.drop('Unnamed: 0', axis=1, inplace=True)
# left_skids = list(bp["left"])
# right_skids = list(bp["right"])

records = bp.to_records(index=False)
skid_pairs = list(records)

#%%

paired, adj = generate_adj_matrix(left_skids, right_skids)
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
# %%
""" bp = pd.read_csv (r"/Users/swilson/Documents/Devneurons/MAnalysis/brain-pairs.csv")
left_skids = list(bp.iloc[:,0])
right_skids = list(bp["right"])

left_names = indexed_names(left_skids)
right_names = indexed_names(right_skids)
name_pairs = list(zip(left_names, right_names))

LRdict = dict(zip(left_names.values(), right_names.values()))
RLdict = dict(zip(right_names.values(), left_names.values()))
 """
# %%
'''just look up partners in that table, make sure its consistent with the skeletons you're pulling with the annotation, 
throw an error if there are any missing or any extra

i'd go through the table and make a {left: right} dict, a {right: left} dict, then go through 
all the skeletons you get with the annotation and make sure they're in exactly one dict
then you can look up the partners easily either way'''