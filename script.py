#%%
from multiprocessing.dummy import freeze_support
import sys
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
        sim = navis.connectivity_similarity(np.array([left_array, right_array]), metric="cosine", n_cores= 1)
        sim_arr = sim.to_numpy()
        val = float(sim_arr[0, 1])
        if np.isnan(val):
            val = None

        out[unsided_name] = val

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    return out
#%%

def merged_adjacency(pair_list) -> pd.DataFrame:
    """ 
    Produce an adjacency matrix where left-right target pairs are each merged into single units.

    The values are input fractions, so merging is done by arithmetic mean (consider harmonic mean?).

    Parameters
    ----------
        pair_list : list of 2-tuples of skeleton IDs

    Returns
    -------
        pd.DataFrame : adjacency matrix whose row labels are skeleton IDs and column labels are indices into the original pair list
    """
    all_skids = []
    for l_r in pair_list:
        all_skids.extend(l_r)
    adj = pymaid.adjacency_matrix(all_skids, fractions=True)
    rows = []
    for _, row in adj.iterrows():
        l, r = row[l_skid], row[r_skid] # are both zero, if so don't add them to row (if sum of both is zero, don't do); if at least one isn't, then add
        rows.append([
            (row[l_skid] + row[r_skid]) / 2 # add by summing, don't divide by two; each row now has a different length, pd might not like
            for l_skid, r_skid in pair_list # rows shouldn't be list, should be dictionary. For every skid (key), value is row constructed from loop (avoiding double zeroes)
        ])

    return pd.DataFrame(rows, index=adj.index) # return two lists (indices correlated); first is adj.index, corresponding row computed above

# leave with all zeroes, in merged analysis (beforecomputing sim) remove all which are zero in both (shorter vector). do this by pair simultaneously
# implement some threshold, e.g. 0.7% ()
# fraction = false, use absolute values for counts ( 2 or less remove)

# prune all double zeroes

def merged_analysis(pair_list):
    """
    Constructs adjacency matrix from merged pairs, scores connectivity (cosine, synaptic threshold of 2)

    Parameters
    ----------
        pair_list : list of 2-tuples of skeleton IDs

    Returns
    -------
        pd.Data Frame: rows = # of pairs (1172), 3 columns (left neuron, right neuron, cosine similarity score)
    """    
   
    adj = merged_adjacency(pair_list)
    sims = []
    for l_skid, r_skid in tqdm(pair_list):
        sim = navis.connectivity_similarity(np.array([adj.loc[l_skid], adj.loc[r_skid]]), "cosine", threshold = 2).to_numpy()[0, 1]
        sims.append(sim)
    
    # for i in range (steps of 2, cos sim for 2 vectors); these 2 may not have same length, 

    df = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
    df["cosine_similarity_merged_targets"] = pd.Series(np.asarray(sims, dtype=float))
    return df

#%%

class HasNextIterator: 
    def __init__(self, it): 
        self._it = iter(it) 
        self._next = None 
 
    def __iter__(self): 
        return self 
 
    def has_next(self): 
        if self._next: 
            return True 
        try: 
            self._next = next(self._it) 
            return True 
        except StopIteration: 
            return False 
 
    def next(self): 
        if self._next: 
            ret = self._next 
            self._next = None 
            return ret 
        elif self.has_next(): 
            return self.next() 
        else: 
            raise StopIteration()


## Run analysis and explore data ##


#%%
METRIC = "cosine"

bp = pd.read_csv(HERE / "brain-pairs.csv")
bp.drop('Unnamed: 0', axis=1, inplace=True)
# left_skids = list(bp["left"])
# right_skids = list(bp["right"])

skid_pairs = bp.to_numpy().astype(int).tolist()

#%%

merged_results = merged_analysis(skid_pairs)
merged_results.to_csv(OUT_DIR / "merged_threshold_cosine_output.tsv", sep="\t", index=False)


sys.exit()

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

''' pair_dict = {}
    for pair in pair_list:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]

    cos_sims = {}
    for pair in pair_list:
        partners = pymaid.get_partners(list(pair))
        # partners = pymaid.get_partners(pair[0])
        # partners.append(pymaid.get_partners(pair[1]))
        partner_skids = set(partners["skeleton_id"])
        columns = []
        while partner_skids:
            skid1 = partner_skids.pop()
            skid2 = pair_dict.get(skid1)
            if skid2 is None:
                # Nothing to fuse. But must normalize
                columns.append(partners[skid1] / float(sum(partners["skid1"])))
            else:
                # Fuse two vectors, normalized
                n_syn1 = float(sum(partners[skid1]))
                n_syn2 = float(sum(partners[skid2]))
                columns.append([v1/n_syn1 + v2/n_syn2 for v1, v2 in zip(partners[skid1], partners[skid2])])
            # Remove both from set
            partner_skids.difference_update([skid1, skid2])
        # Two rows, as many colums as were needed
        matrix = np.array(columns).transpose() # rows have to be skid1, skid2

        score = navis.connectivity_similarity(matrix, metric="cosine")
        cos_sims[skid1] = score[0]
        cos_sims[skid2] = score[1]

    # Save the similarity scores
    # TODO create panda DataFrame with int column for skids and float column for scores, and save as CSV

        out = pd.DataFrame(columns = ["skid_left", "skid_right", "score"])
'''

def fusion(pair_list):
    # list_pairs: a list of tuples, assumed to be skeleton IDs
    
    pair_dict = {}
    for pair in pair_list:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    
    cos_sims = {}
    for pair in pair_list:
        partners = pymaid.get_partners(list(pair))    
        partner_skids = set(partners["skeleton_id"])
        columns = []
        itr = iter(partner_skids)
        while partner_skids:
            skid1 = partner_skids.pop()
            skid2 = pair_dict.get(skid1)
            if skid2 is None:
                # Nothing to fuse. But must normalize
                columns.append(partners["skid1"] / float(sum(partners["skid1"])))
            else:
                # Fuse two vectors, normalized
                n_syn1 = float(sum(partners["skid1"]))
                n_syn2 = float(sum(partners["skid2"]))
                columns.append([v1/n_syn1 + v2/n_syn2 for v1, v2 in zip(partners["skid1"], partners["skid2"])])
            # Remove both from set
            partner_skids.difference_update([skid1, skid2])
        # Two rows, as many colums as were needed
        matrix = np.array(columns).transpose() # rows have to be skid1, skid2

        score = navis.connectivity_similarity(matrix, metric="cosine")
        cos_sims[skid1] = score[0]
        cos_sims[skid2] = score[1]

    # Save the similarity scores
    # TODO create panda DataFrame with int column for skids and float column for scores, and save as CSV
            
        out = pd.DataFrame(columns = ["skid_left", "skid_right", "score"])
        
