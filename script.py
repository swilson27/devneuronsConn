#!/usr/bin/env python

#%%
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
from pyrsistent import T
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
'''from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent'''

import json
with open("seymour.json") as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)



### LR connectivity analysis ###



## Define functions


#%%
def get_neurons():
    ''' get CatmaidNeuronLists of left and right brain pairs.
        N.B. lists are each in arbitrary order '''
    fpath = "Cneurons.pickle"
    if not os.path.exists(fpath):
        import pymaid
        neurons = tuple(pymaid.get_neuron("annotation:sw;brainpair;" + side) for side in "LR")

        with open("Cneurons.pickle", "wb") as f:
            pickle.dump(neurons, f, protocol=5)
    else:
        with open("Cneurons.pickle", "rb") as f:
            neurons = pickle.load(f)

    return neurons


def name_pair(left_names, right_names):
    ''' assign tags to pairs'''

    unsided_l = {n.replace("left", "__SIDE__"): n for n in left_names}
    unsided_r = {n.replace("right", "__SIDE__"): n for n in right_names}
    in_both = set(unsided_l).intersection(unsided_r)
    for unsided_name in sorted(in_both):
        yield (unsided_l[unsided_name], unsided_r[unsided_name])

#%%
def generate_adj_matrix():
    ''' generate and load adjacency matrix for each pair '''

    neurons_l, neurons_r = get_neurons()
    by_name_l = dict(zip(neurons_l.name, neurons_l))
    by_name_r = dict(zip(neurons_r.name, neurons_r))
    paired = list(name_pair(by_name_l, by_name_r))
    paireDF = pd.DataFrame(paired)
    paireDF = paireDF.iloc[0:10,]

    '''below is where the issues are. I figured I could just get respective
    connectivity tables for each L/R pair, and then iterate through to generate
    an adjacency matrix (mat) row by row for these pairs. This list of adj. matrices
    would then be used in a following function to compute a connectivity score for each pair,
    and store this all in a DF

    Unfortunately this generation of 'mat' doesn't work quite as hoped; I ended up commenting out, and running block by block, to isolate
    the problem. Now think perhaps my plan was erroneous, and that I should have tried to generate a connectivity score within this 
    iteration block? (opposed to inputting mat into a separate function). Any thoughts would be great; I'll try now to run this all in one
    loop, just would be very interested to see how you'd approach and code functions for these goals (to generate connectivity similarity
    scores for each LR pair (p.s. I am completely aware I'd need to merge LR homologues (see bottom of script), just wished to get this bit 
    working first'''

    mat = []
    for row in paireDF.itertuples():
        cn_table_L = pymaid.get_partners(row[0])
        cn_table_R = pymaid.get_partners(row[1])

        mat.append(pymaid.adjacency_matrix(cn_table_L, cn_table_R,
        use_connectors=True, fractions=True))
        print(row)
    return(mat)

'''this code below was the initial template for a results data frame I was hoping to generate (3 columns, rows: # of pairs)
Was intending to use a separate function, but now think it may be best to wrap it all in the previous one'''

'''results_conn = pd.DataFrame(columns=["left_neuron", "right_neuron", "connectivity_score"])
for pair, l_name, r_name in adjacency:
    navis.connectivity_similarity(adjacency, metric='vertex_normalized', threshold=3)
    results_conn.loc[pair] = [adjacency[by_name_l]]'''

    '''count = (adj.iloc[:,2] != 0).sum()
    print(count)'''

'''this block below is just where I considered switching paireDF's representation of the pairs from the neuron names to skids'''

#%%
neurons_l, neurons_r = get_neurons()
by_name_l = dict(zip(neurons_l.name, neurons_l))
by_name_r = dict(zip(neurons_r.name, neurons_r))
paired = list(name_pair(by_name_l, by_name_r))
paireDF = pd.DataFrame(paired)
paireDF = paireDF.iloc[0:10,]

for row in paireDF.apply():
    pymaid.get_skids_by_name(paireDF)



#%%

'''Code from Albert, which effectively merges L-R homologues. I should be able to adapt and use it for this project'''

from lib.util.csv import parseLabeledMatrix
from lib.util.matrix import combineConsecutivePairs

def mergeNormalized(matrix_csv_path, measurements_csv_path, fix, single, joint):
    """
    Load the matrix in the CSV path,
    normalize it relative to the inputs listed in the measurements CSV path,
    and merge the left-right homologous pairs in the matrix_csv_path
    according to rules involving the single and joint thresholds of synaptic counts,
    and normalizing by the total amount of inputs as present in the measurements_csv_path.
    'fix': (optional, can be None) a function that receives 3 arguments: row_names, column_names and matrix, and returns the same 3 arguments, fixing ordering etc. as necessary.
    'single': synapse count threshold for an individual connection. 3 can be a reasonable value.
    'joint': synapse count threshold for a joint left-right homologous pair of connections. 10 can be a reasonable value.
    """

    if not fix:
        def fix(*args):
            return args
    # Load
    row_names, column_names, matrix = fix(*parseLabeledMatrix(matrix_csv_path, cast=float, separator=','))

    # Load the table of measurements
    neuron_names, measurement_names, measurements = parseLabeledMatrix(measurements_csv_path, cast=float, separator=',')
    # Create map of neuron name vs amount of postsynaptic sites in its arbor
    # Index 4 (column 4) of measurements is "N inputs"
    n_inputs = {name: row[4] for name, row in zip(neuron_names, measurements)}

    # Normalize: divide each value by the total number of postsynaptic sites in the entire postsynaptic neuron.
    normalized = []
    for row in matrix:
        normalized.append([a / n_inputs[name] for name, a in zip(column_names, row)])

    # Merge left and right partners:
    # Requires both left and right having at least 3 synapses, and the sum at least 10
    # Uses the matrix of synapse counts to filter, but returns values from the normalized matrix.
    def mergeFn(matrix, rowIndex1, rowIndex2, colIndex1, colIndex2):
        row11 = matrix[rowIndex1][colIndex1]
        row12 = matrix[rowIndex1][colIndex2]
        row21 = matrix[rowIndex2][colIndex1]
        row22 = matrix[rowIndex2][colIndex2]
        if (row11 >= single or row12 >= single) and (row21 >= single or row22 >= single) and row11 + row12 + row21 + row22 >= joint:
            n11 = normalized[rowIndex1][colIndex1]
            n12 = normalized[rowIndex1][colIndex2]
            n21 = normalized[rowIndex2][colIndex1]
            n22 = normalized[rowIndex2][colIndex2]
            s = 0.0
            count = 0
            for n_syn, norm in zip([row11, row12, row21, row22], \
                                   [  n11,   n12,   n21,   n22]):
                if n_syn >= single:
                    s += norm
                    count += 1
            return (s / count) * 100 if count > 0 else 0
        return 0

    combined = combineConsecutivePairs(matrix, aggregateFn=mergeFn, withIndices=True)

    # Merge names: remove the " LEFT" and " RIGHT" suffixes
    row_names = [name[:name.rfind(' ')] for name in row_names[::2]]
    column_names = [name[:name.rfind(' ')] for name in column_names[::2]]

    return combined, row_names, column_names