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

#pymaid.connect_catmaid()


#DEFAULT_SEED = 1991



### LR connectivity analysis ###



## Define functions


#%%
def get_neurons():
    ''' get CatmaidNeuronLists of left and right brain pairs.
        N.B. lists are each in arbitrary order '''
    fpath = "Cneurons.pickle"
    if not os.path.exists(fpath):
        import pymaid

        rm = pymaid.CatmaidInstance()


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

    '''here's where the issues are. I figured I could just get respective
    connectivity tables for each L/R pair, and then iterate through to generate
    an adjacency matrix (and potentially a connectivity similarity
    assessment/quantification) row by row

    I ended up commenting out, and running block by block, to isolate
    the problem; it seems this part doesn't work as hoped. Perhaps I
    should have tried to generate a connectivity score within this iteration?'''

    mat = []
    for row in paireDF.itertuples():
        cn_table_L = pymaid.get_partners(row[0])
        cn_table_R = pymaid.get_partners(row[1])

        mat.append(pymaid.adjacency_matrix(cn_table_L, cn_table_R,
        use_connectors=True, fractions=True))
        print(row)
    return(mat)'''
    print(cn_table_L)
    print(cn_table_R)

    '''results_conn = pd.DataFrame(columns=["left_neuron", "right_neuron", "connectivity_score"])
    for pair, l_name, r_name in adjacency:
        navis.connectivity_similarity(adjacency, metric='vertex_normalized', threshold=3)
        results_conn.loc[pair] = [adjacency[by_name_l]]'''

    '''count = (adj.iloc[:,2] != 0).sum()
    print(count)'''

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

## Apply analysis to neuron lists

