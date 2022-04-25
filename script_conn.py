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
from itertools import chain


logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "output"
# Logging tracker of events/errors; returns parent directory of given path, provides cache and output directories

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
# Loads stored catmaid credentials



### LR connectivity analysis ###



#%%
METRIC = "cosine"

bp = pd.read_csv(HERE / "brain-pairs.csv")
subsetbp = bp[bp.region.isna()]
subsetbp.drop('region', axis=1, inplace=True)
skid_pairs = subsetbp.to_numpy().astype(int).tolist()
bp.drop('region', axis=1, inplace=True)
brain_pairs = bp.to_numpy().astype(int).tolist()


## Define functions


#%%
def merged_analysis(pair_list, metric = "cosine", is_input = False):
    """
    Constructs adjacency matrix ('columns') from merged pairs (normalized sum), 
    scores connectivity similarity (cosine, synaptic threshold of 2) between these pair members, and outputs scores as csv

    Partners is a DF, with all of pairs connecting neurons as rows and columns: name, skid, relation, total n_syn, pair[0] n_syn, pair[1] n_syn


    Parameters
    ----------
        pair_list : nested list of skeleton IDs

    Returns
    -------
        pd.Data Frame: rows = # of pairs (1172), 3 columns (left neuron, right neuron, cosine similarity score)
    """    
    # out_file = OUT_DIR / f"canalysis_{metric}_{'in' if is_input else 'out'}put.json"
    # if out_file.is_file():
    #     with open(out_file) as f:
    #         return json.load(f)

    pair_dict = {}
    for skids in brain_pairs:
        pair_dict[skids[0]] = skids[1]
        pair_dict[skids[1]] = skids[0]
    # Construct dictionary of all brain pairs (from CSV)


    sims = []
    # For each pair iteratively, obtain all partners and filter to get just input or outputs (i/o)
    for pair in pair_list:
        partners = pymaid.get_partners(list(pair))
        if is_input == True:
            partners = partners[partners.relation == 'upstream']
        else: 
            partners = partners[partners.relation == 'downstream']

        partner_skids = set(partners["skeleton_id"])

        sum0 = float(sum(partners.iloc[:, 4]))
        if sum0 == 0:
            sims.append(np.NaN)
        sum1 = float(sum(partners.iloc[:, 5]))
        if sum1 == 0:
            sims.append(np.NaN)
        # If a sum of synaptic counts for pair's i/o connections is 0, return Na for connectivity score
        if sum0 and sum1 != 0:

            column0 = []
            column1 = []
            # Iteratively go through partners, and apply merging and normalization protocol
            while partner_skids:
                partner = partner_skids.pop()
                partner_homolog = pair_dict.get(partner) # Partner_homolog is the L-R pair of partner
                
                if partner_homolog is None:
                    # No L-R pair to merge, but must normalize (partner:L/R-pair synapses ÷ sum L/R synapses)
                    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] / sum0
                    column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
                    # try:
                    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
                    # except ZeroDivisionError:
                    #     col1 = 0
                    column1.append(pd.to_numeric(col1.values, errors = 'coerce'))
                else:
                    # Merge the two vectors (L-R pairs) and normalize ()
                    col0 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])].values) / sum0
                    column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
                    col1 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])].values) / sum1
                    column1.append(pd.to_numeric(col1.values, errors = 'coerce'))
                
                # Remove both from set, before iterating to next partner
                partner_skids.discard(partner)
                partner_skids.discard(partner_homolog)   

            # Obtain columns of merged/normalized values (initially as ndarray of 1-item ndarrays) and convert into adjacency matrix
            column0 = pd.Series(column0).values
            column1 = pd.Series(column1).values

            flat_col0 = np.array(list(chain.from_iterable(column0)))
            flat_col1 = np.array(list(chain.from_iterable(column1)))
            matrix = pd.DataFrame([flat_col0, flat_col1]) # rows have to be pair_L, pair_R

            # Calculate connectivity score for pair and append 
            scores = navis.connectivity_similarity(adjacency = matrix, metric = metric, n_cores= 1)
            sims.append(scores.iloc[0][1])

    print(sims)
            
        # Output scores as pd.DF, int columns for skids & float for scores 
    out = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
    out["cosine_similarity"] = pd.Series(np.asarray(sims, dtype=float))

    # with open(out_file, "w") as f:
    #      out_json = out.to_json()
    #      json.dump(out_json, f, indent=2, sort_keys=True)

    return out
    


## Run analysis and explore data ##


#%%
METRIC = "cosine"

bp = pd.read_csv(HERE / "brain-pairs.csv")
bp = bp[bp.region.isna()]
bp.drop('region', axis=1, inplace=True)
skid_pairs = bp.to_numpy().astype(int).tolist()

#%%
merged_input = merged_analysis(skid_pairs, is_input = True)
merged_input.to_csv(OUT_DIR / "merged_threshold_cosine_input.csv", index=False)

#%%
merged_output = merged_analysis(skid_pairs, is_input = False)
merged_output.to_csv(OUT_DIR / "merged_threshold_cosine_output.csv", index=False)


sys.exit()

## run for specific pair
partners = pymaid.get_partners(list(pair))
partners = partners[partners.relation == 'upstream']
partner_skids = set(partners["skeleton_id"])
sum0 = float(sum(partners.iloc[:, 4]))
sum1 = float(sum(partners.iloc[:, 5]))
column0 = []
column1 = []
while partner_skids:
    partner = partner_skids.pop()
    partner_homolog = pair_dict.get(partner) 

    if partner_homolog is None:
        col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] / sum0
        column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
        col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
        column1.append(pd.to_numeric(col1.values, errors = 'coerce'))
    else:
        col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])] / sum0
        column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
        col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])] / sum1
        column1.append(pd.to_numeric(col1.values, errors = 'coerce'))

    partner_skids.discard(partner)
    partner_skids.discard(partner_homolog) 

column0 = pd.Series(column0).values
column1 = pd.Series(column1).values
flat_col0 = np.array(list(chain.from_iterable(column0)))
flat_col1 = np.array(list(chain.from_iterable(column1)))
matrix = pd.DataFrame([flat_col0, flat_col1])
print(matrix)
score = navis.connectivity_similarity(adjacency = matrix, metric = 'cosine', n_cores= 1)
score = score.iloc[0][1]
print(score)


## Specific col calculations

partner = 
partner_homolog = pair_dict.get(partner)
                
if partner_homolog is None:
    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] / sum0
    column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
    column1.append(pd.to_numeric(col1.values, errors = 'coerce'))
else:
    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])] / sum0
    column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])] / sum1
    column1.append(pd.to_numeric(col1.values, errors = 'coerce'))


## old analysis code to revamp

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
