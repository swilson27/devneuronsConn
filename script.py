#!/usr/bin/env python

#%%
from doctest import NORMALIZE_WHITESPACE
from black import out
import pymaid
import numpy as np
import navis
from multiprocessing import Pool, freeze_support
from copy import deepcopy
import itertools
import os
import pickle
from collections import Counter
from tqdm import tqdm
from ipywidgets import IntProgress
import csv

# %%
import logging
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar

#%%
import navis
import numpy as np
import pandas as pd
import pymaid

import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "output"

SIDE_PLACEHOLDER = "__SIDE__"

import json

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)



### LR connectivity analysis ###



## Define functions


#%%
def get_neurons():
    ''' get CatmaidNeuronLists of left and right brain pairs.
        N.B. lists are each in arbitrary order '''
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


def name_pair(left_neurons, right_neurons):
    ''' assign tags to pairs'''

    unsided_l = {n.name.replace("left", SIDE_PLACEHOLDER): n for n in left_neurons if "left" in n.name}
    unsided_r = {n.name.replace("right", SIDE_PLACEHOLDER): n for n in right_neurons if "right" in n.name}
    in_both = set(unsided_l).intersection(unsided_r)
    for unsided_name in sorted(in_both):
        yield unsided_name, (unsided_l[unsided_name], unsided_r[unsided_name])


#%%
def generate_adj_matrix():
    ''' generate and load adjacency matrix for each pair '''

    neurons_l, neurons_r = get_neurons()
    paired = dict(name_pair(neurons_l, neurons_r))

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
        out[unsided_name] = float(sim_arr[0, 1])

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    return out


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


# print(adj)


    # paireDF = pd.DataFrame(paired)
    # paireDF = paireDF.iloc[0:10,]

    # below is where the issues are. I figured I could just get respective
    # connectivity tables for each L/R pair, and then iterate through to generate
    # an adjacency matrix (mat) row by row for these pairs. This list of adj. matrices
    # would then be used in a following function to compute a connectivity score for each pair,
    # and store this all in a DF

    # Unfortunately this generation of 'mat' doesn't work quite as hoped; I ended up commenting out, and running block by block, to isolate
    # the problem. Now think perhaps my plan was erroneous, and that I should have tried to generate a connectivity score within this
    # iteration block? (opposed to inputting mat into a separate function). Any thoughts would be great; I'll try now to run this all in one
    # loop, just would be very interested to see how you'd approach and code functions for these goals (to generate connectivity similarity
    # scores for each LR pair (p.s. I am completely aware I'd need to merge LR homologues (see bottom of script), just wished to get this bit
    # working first

    # mat = []
    # for row in paireDF.itertuples():
    #     cn_table_L = pymaid.get_partners(row[0])
    #     cn_table_R = pymaid.get_partners(row[1])

    #     mat.append(pymaid.adjacency_matrix(cn_table_L, cn_table_R,
    #     use_connectors=True, fractions=True))
    #     print(row)
    # return(mat)

# this code below was the initial template for a results data frame I was hoping to generate (3 columns, rows: # of pairs)
# Was intending to use a separate function, but now think it may be best to wrap it all in the previous one

# results_conn = pd.DataFrame(columns=["left_neuron", "right_neuron", "connectivity_score"])
# for pair, l_name, r_name in adjacency:
#     navis.connectivity_similarity(adjacency, metric='vertex_normalized', threshold=3)
#     results_conn.loc[pair] = [adjacency[by_name_l]]

    # count = (adj.iloc[:,2] != 0).sum()
    # print(count)

# this block below is just where I considered switching paireDF's representation of the pairs from the neuron names to skids

#%%
# neurons_l, neurons_r = get_neurons()
# by_name_l = dict(zip(neurons_l.name, neurons_l))
# by_name_r = dict(zip(neurons_r.name, neurons_r))
# paired = list(name_pair(by_name_l, by_name_r))
# paireDF = pd.DataFrame(paired)
# paireDF = paireDF.iloc[0:10,]

# for row in paireDF.apply():
#     pymaid.get_skids_by_name(paireDF)



#%%

# '''Code from Albert, which effectively merges L-R homologues. I should be able to adapt and use it for this project'''

# from lib.util.csv import parseLabeledMatrix
# from lib.util.matrix import combineConsecutivePairs

# def mergeNormalized(matrix_csv_path, measurements_csv_path, fix, single, joint):
#     """
#     Load the matrix in the CSV path,
#     normalize it relative to the inputs listed in the measurements CSV path,
#     and merge the left-right homologous pairs in the matrix_csv_path
#     according to rules involving the single and joint thresholds of synaptic counts,
#     and normalizing by the total amount of inputs as present in the measurements_csv_path.
#     'fix': (optional, can be None) a function that receives 3 arguments: row_names, column_names and matrix, and returns the same 3 arguments, fixing ordering etc. as necessary.
#     'single': synapse count threshold for an individual connection. 3 can be a reasonable value.
#     'joint': synapse count threshold for a joint left-right homologous pair of connections. 10 can be a reasonable value.
#     """

#     if not fix:
#         def fix(*args):
#             return args
#     # Load
#     row_names, column_names, matrix = fix(*parseLabeledMatrix(matrix_csv_path, cast=float, separator=','))

#     # Load the table of measurements
#     neuron_names, measurement_names, measurements = parseLabeledMatrix(measurements_csv_path, cast=float, separator=',')
#     # Create map of neuron name vs amount of postsynaptic sites in its arbor
#     # Index 4 (column 4) of measurements is "N inputs"
#     n_inputs = {name: row[4] for name, row in zip(neuron_names, measurements)}

#     # Normalize: divide each value by the total number of postsynaptic sites in the entire postsynaptic neuron.
#     normalized = []
#     for row in matrix:
#         normalized.append([a / n_inputs[name] for name, a in zip(column_names, row)])

#     # Merge left and right partners:
#     # Requires both left and right having at least 3 synapses, and the sum at least 10
#     # Uses the matrix of synapse counts to filter, but returns values from the normalized matrix.
#     def mergeFn(matrix, rowIndex1, rowIndex2, colIndex1, colIndex2):
#         row11 = matrix[rowIndex1][colIndex1]
#         row12 = matrix[rowIndex1][colIndex2]
#         row21 = matrix[rowIndex2][colIndex1]
#         row22 = matrix[rowIndex2][colIndex2]
#         if (row11 >= single or row12 >= single) and (row21 >= single or row22 >= single) and row11 + row12 + row21 + row22 >= joint:
#             n11 = normalized[rowIndex1][colIndex1]
#             n12 = normalized[rowIndex1][colIndex2]
#             n21 = normalized[rowIndex2][colIndex1]
#             n22 = normalized[rowIndex2][colIndex2]
#             s = 0.0
#             count = 0
#             for n_syn, norm in zip([row11, row12, row21, row22], \
#                                    [  n11,   n12,   n21,   n22]):
#                 if n_syn >= single:
#                     s += norm
#                     count += 1
#             return (s / count) * 100 if count > 0 else 0
#         return 0

#     combined = combineConsecutivePairs(matrix, aggregateFn=mergeFn, withIndices=True)

#     # Merge names: remove the " LEFT" and " RIGHT" suffixes
#     row_names = [name[:name.rfind(' ')] for name in row_names[::2]]
#     column_names = [name[:name.rfind(' ')] for name in column_names[::2]]

#     return combined, row_names, column_names
