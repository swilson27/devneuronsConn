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
# Logging tracker of events/errors; returns parent directory of given path, provides cache and output directories

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
# Loads stored catmaid credentials



### LR connectivity analysis ###



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
    out_file = OUT_DIR / f"canalysis_{metric}_{'in' if is_input else 'out'}put.json"
    if out_file.is_file():
        with open(out_file) as f:
            return json.load(f)

    pair_dict = {}
    for pair in pair_list:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    
    sims = []
    for pair in pair_list:
        partners = pymaid.get_partners(list(pair))
        if is_input == True:
            partners = partners[partners.relation == 'upstream']
        else: 
            partners = partners[partners.relation == 'downstream']

        partner_skids = set(partners["skeleton_id"])

        sum0 = float(sum(partners.iloc[:, 4]))
        sum1 = float(sum(partners.iloc[:, 5]))

        column0 = []
        column1 = []
        while partner_skids:
            partner = partner_skids.pop()
            partner_homolog = pair_dict.get(partner) #partner_homolog is the L-R pair of partner

            """ if partner_homolog is None:
                # Nothing to fuse, but must normalize
                col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values / sum0
                column0.append(pd.to_numeric(col0, errors = 'coerce'))
                col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values / sum1
                column1.append(pd.to_numeric(col1, errors = 'coerce'))
                print(type(col0))
                print(type(col1))
            else:
                # Fuse two vectors, normalized
                col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])].values / sum0
                column0.append(pd.to_numeric(col0, errors = 'coerce'))
                col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])].values / sum1
                column1.append(pd.to_numeric(col1, errors = 'coerce')) """
            
            if partner_homolog is None:
                # Nothing to fuse, but must normalize
                col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] / sum0
                column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
                # try:
                col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
                # except ZeroDivisionError:
                #     col1 = 0
                column1.append(pd.to_numeric(col1, errors = 'coerce'))
            else:
                # Fuse two vectors, normalized
                col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])] / sum0
                column0.append(pd.to_numeric(col0, errors = 'coerce'))
                col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])] / sum1
                column1.append(pd.to_numeric(col1, errors = 'coerce'))
            
            # Remove both from set
            partner_skids.discard(partner)
            partner_skids.discard(partner_homolog)          
        # Two rows, as many columns as were needed
        column0 = pd.Series(column0).values
        column1 = pd.Series(column1).values

        flat_col0 = []
        for sublist in column0:
            for item in sublist:
                flat_col0.append(item)

        flat_col1 = []
        for sublist in column1:
            for item in sublist:
                flat_col1.append(item)

        matrix = pd.DataFrame([flat_col0, flat_col1]) # rows have to be pair_L, pair_R
        scores = navis.connectivity_similarity(adjacency = matrix, metric = metric, n_cores= 1)
        sims.append(scores.iloc[0][1])
    
    print(sims)


METRIC = "cosine"

bp = pd.read_csv(HERE / "brain-pairs.csv")
bp.drop('Unnamed: 0', axis=1, inplace=True)
# left_skids = list(bp["left"])
# right_skids = list(bp["right"])

skid_pairs = bp.to_numpy().astype(int).tolist()

#%%
merged_input = merged_analysis(skid_pairs, is_input = True)

print(merged_input)

sys.exit()

#%%
    # Output scores as pd.DF, int columns for skids & float for scores 
    # out = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
    # out["cosine_similarity"] = pd.Series(np.asarray(sims, dtype=float))

    # with open(out_file, "w") as f:
    #     out_json = out.to_json()
    #     json.dump(out_json, f, indent=2, sort_keys=True)

    # return out


## Run analysis and explore data ##


#%%
METRIC = "cosine"

bp = pd.read_csv(HERE / "brain-pairs.csv")
bp.drop('Unnamed: 0', axis=1, inplace=True)
# left_skids = list(bp["left"])
# right_skids = list(bp["right"])

skid_pairs = bp.to_numpy().astype(int).tolist()

#%%
merged_input = merged_analysis(skid_pairs, is_input = True)
print(merged_input)

merged_input.to_csv(OUT_DIR / "merged_threshold_cosine_input.csv", index=False)

#%%
merged_output = merged_analysis(skid_pairs, is_input = False)
merged_output.to_csv(OUT_DIR / "merged_threshold_cosine_output.csv", index=False)


sys.exit()

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
# %%

pair_dict = {}
for pair in skid_pairs:
    pair_dict[pair[0]] = pair[1]
    pair_dict[pair[1]] = pair[0]

sims = []
for pair in skid_pairs[0:10]:
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
            try:
                col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
            except ZeroDivisionError:
                col1 = 0
                column1.append(pd.to_numeric(col1, errors = 'coerce'))
        else:
                # Fuse two vectors, normalized
            col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])] / sum0
            column0.append(pd.to_numeric(col0, errors = 'coerce'))
            col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])] / sum1
            column1.append(pd.to_numeric(col1, errors = 'coerce'))
            
            # Remove both from set
        partner_skids.discard(partner)
        partner_skids.discard(partner_homolog)          
        # Two rows, as many columns as were needed
    column0 = pd.Series(column0).values
    column1 = pd.Series(column1).values

    flat_col0 = []
    for sublist in column0:
        for item in sublist:
            flat_col0.append(item)

    flat_col1 = []
    for sublist in column1:
        for item in sublist:
            flat_col1.append(item)
        

    matrix = pd.DataFrame([flat_col0, flat_col1]) # rows have to be pair_L, pair_R
    print(matrix)
        # # rows have to be pair_L, pair_R
    if __name__ == '__main__':
        sims = navis.connectivity_similarity(adjacency = matrix, metric = "cosine", threshold = 2)
        sims.append(sims)


print(sims)

    # Output scores as pd.DF, int columns for skids & float for scores 
out = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
out["cosine_similarity"] = pd.Series(np.asarray(sims, dtype=float))



# %%
