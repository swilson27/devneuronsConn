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
from pytz import ZERO
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
# Logging tracker of events/errors; returns parent directory of given path, provides cache and output directories for folder

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
# Loads stored catmaid credentials of individual (file named as above; must be a json containing server, api_token, http_user and http_password variables [as described on pymaid website])



### LR connectivity analysis ###



## Define functions


#%%
def merged_analysis(pair_list, metric = "cosine", is_input = False, threshold = 2):
    """
    Constructs adjacency matrix ('columns') from merged pairs (normalized sum), 
    scores connectivity similarity (default as cosine, synaptic threshold of 2 [summed across L-R pair]) between these pair members, and outputs scores as csv

    Partners is a DF, with all of pairs connecting neurons as rows and columns: name, skid, relation, total n_syn, pair[0] n_syn, pair[1] n_syn


    Parameters
    ----------
        pair_list : nested list of skeleton IDs

    Returns
    -------
        pd.Data Frame: rows = # of pairs (1172), 3 columns (left neuron, right neuron, connectivity similarity score)
    """    
    # If already computed, load as json file
    # out_file = OUT_DIR / f"merged_{metric}_th{threshold}_{'in' if is_input else 'out'}put.json"
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
        partners = pymaid.get_partners(list(pair), threshold = threshold)
        if is_input == True:
            partners = partners[partners.relation == 'upstream']
        else: 
            partners = partners[partners.relation == 'downstream']
        sum0 = float(sum(partners.iloc[:, 4]))
        sum1 = float(sum(partners.iloc[:, 5]))
        if sum0 == 0 or sum1 == 0:
            sims.append(np.NaN)            
        # If a sum of synaptic counts for pair's i/o connections is 0, return Na for connectivity score (no comparison to be made)

        if sum0 and sum1 != 0:
        # If not, compute merged connectivity similarity analysis
            column0 = []
            column1 = []
            partner_skids = set(partners["skeleton_id"])
            # Iteratively go through partners, and apply merging and normalization protocol
            while partner_skids:
                partner = partner_skids.pop()
                partner_homolog = pair_dict.get(int(partner)) # Partner_homolog is the L-R pair of partner
                
                if partner_homolog is None:
                    # No L-R pair to merge, but must normalize (partner:L/R-pair synapses ÷ sum L/R synapses)
                    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values / sum0
                    column0.append(col0)
                    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values / sum1
                    column1.append(col1)
                else:
                    # Remove from next iteration
                    partner_skids.discard(partner_homolog)
                    # Merge the two vectors (L-R pairs) and normalize ()
                    col0 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])].values) / sum0
                    column0.append(col0)
                    col1 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])].values) / sum1
                    column1.append(col1)

            # Obtain columns of merged/normalized values (initially as ndarray of 1-item ndarrays) and convert into adjacency matrix
            flat_col0 = np.array(list(chain.from_iterable(column0)))
            flat_col1 = np.array(list(chain.from_iterable(column1)))
            matrix = pd.DataFrame([flat_col0, flat_col1])

            # # Calculate connectivity score for pair and append to sims
            score = navis.connectivity_similarity(adjacency = matrix, metric = metric, n_cores= 1)
            sims.append(score.iloc[0][1])
            
    # Output scores as pd.DF, int columns for skids & float for scores 
    out = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
    out[f"{metric}_similarity"] = pd.Series(np.asarray(sims, dtype=float))

    # Store as json for future
    # with open(out_file, "w") as f:
    #     out_json = out.to_json()
    #     json.dump(out_json, f, indent=2, sort_keys=True)

    return out  


def csv_export(pair_list, metric = "cosine", threshold = 2):
    """ Take merged_analysis results (either as pickled json or by running analysis) and export as CSV, adding column of neuron names

    Args: all same types as merged_analysis

    Returns:
        :  2 CSVs, for analysis of both input and output connections separately
           4 identical column names for both: left_skid, right_skid, connectivity metric and left_names
    """    

    # Input conversion
    input = merged_analysis(pair_list, is_input = True)
    print(type(input))
    print(input)
    input = input.sort_values('left_skid')
    input = input.reset_index(drop = True)

    # Get names of all left_skids and assign to new column
    vals = pymaid.get_names(list(input['left_skid']))
    vals = list(vals.values())
    # CSV contains 2 duplicates, must insert these names to compensate for order
    vals.insert(253, 'AVL011 PN Right 2?')
    vals.insert(514, 'OLP4;right')
    # 154 and 341 respectively, if using skid_pairs
    input["left_name"] = pd.Series(vals)
    input = input.sort_values('cosine_similarity')

    # Export to csv
    input.to_csv(OUT_DIR / f"canalysis_{metric}_th{threshold}_input.csv", index=False)

    # Output conversion (virtually identical code)
    output = merged_analysis(pair_list, is_input = False)
    output = output.sort_values('left_skid')
    output = output.reset_index(drop = True)
    output["left_name"] = pd.Series(vals)
    output = output.sort_values('cosine_similarity')
    output.to_csv(OUT_DIR / f"canalysis_{metric}_th{threshold}_output.csv", index=False)

    return input, output


## Run analysis ##


#%%
# Read CSV of all brain pairs (three columns: left and right pairs as skids, also annotated region)
bp = pd.read_csv(HERE / "brain-pairs.csv")
subsetbp = bp[bp.region.isna()]
subsetbp.drop('region', axis=1, inplace=True)
skid_pairs = subsetbp.to_numpy().astype(int).tolist()
bp.drop('region', axis=1, inplace=True)
brain_pairs = bp.to_numpy().astype(int).tolist()
# Brain pairs represents all of these as a nested list of skids; skid_pairs is a subset, containing only original list (which was not annotated)

#%%
# Run functions and export as csv. No need to specify metric or threshold if using defaults (cosine and 2 respectively)
input, output = csv_export(brain_pairs)


## Specific functions for input/output analysis directly, or isolated pairs and partners


#%%
def merged_input(pair_list, metric = "cosine"):
    pair_dict = {}
    for skids in brain_pairs:
        pair_dict[skids[0]] = skids[1]
        pair_dict[skids[1]] = skids[0]
    # Construct dictionary of all brain pairs (from CSV)

    sims = []
    # For each pair iteratively, obtain all partners and filter to get just input or outputs (i/o)
    for pair in pair_list:
        partners = pymaid.get_partners(list(pair), directions = ['incoming'], threshold = 2)
        sum0 = float(sum(partners.iloc[:, 4]))
        sum1 = float(sum(partners.iloc[:, 5]))
        print(sum0, sum1)
        if sum0 == 0 or sum1 == 0:
            sims.append(np.NaN)
        if sum0 and sum1 != 0:
            column0 = []
            column1 = []
            partner_skids = set(partners["skeleton_id"])
            # Iteratively go through partners, and apply merging and normalization protocol
            while partner_skids:
                partner = partner_skids.pop()
                partner_homolog = pair_dict.get(int(partner)) # Partner_homolog is the L-R pair of partner
                
                if partner_homolog is None:
                    # No L-R pair to merge, but must normalize (partner:L/R-pair synapses ÷ sum L/R synapses)
                    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values / sum0
                    column0.append(col0)
                    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values / sum1
                    column1.append(col1)
                else:
                    # Remove from next iteration
                    partner_skids.discard(partner_homolog)
                    # Merge the two vectors (L-R pairs) and normalize ()
                    col0 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])].values) / sum0
                    column0.append(col0)
                    col1 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])].values) / sum1
                    column1.append(col1)

            # Obtain columns of merged/normalized values (initially as ndarray of 1-item ndarrays) and convert into adjacency matrix
            flat_col0 = np.array(list(chain.from_iterable(column0)))
            flat_col1 = np.array(list(chain.from_iterable(column1)))
            matrix = pd.DataFrame([flat_col0, flat_col1])

            # # Calculate connectivity score for pair and append to sims
            score = navis.connectivity_similarity(adjacency = matrix, metric = metric, n_cores= 1)
            sims.append(score.iloc[0][1])
            # print(sims)
            
    # Output scores as pd.DF, int columns for skids & float for scores 
    out = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
    out["cosine_similarity"] = pd.Series(np.asarray(sims, dtype=float))

    return out  

def merged_output(pair_list, metric = "cosine"):
    pair_dict = {}
    for skids in brain_pairs:
        pair_dict[skids[0]] = skids[1]
        pair_dict[skids[1]] = skids[0]
    # Construct dictionary of all brain pairs (from CSV)

    sims = []
    # For each pair iteratively, obtain all partners and filter to get just input or outputs (i/o)
    for pair in pair_list:
        partners = pymaid.get_partners(list(pair), directions = ['outgoing'], threshold = 2)
        sum0 = float(sum(partners.iloc[:, 4]))
        sum1 = float(sum(partners.iloc[:, 5]))
        if sum0 == 0 or sum1 == 0:
            sims.append(np.NaN)
        if sum0 and sum1 != 0:
            column0 = []
            column1 = []
            partner_skids = set(partners["skeleton_id"])
            # Iteratively go through partners, and apply merging and normalization protocol
            while partner_skids:
                partner = partner_skids.pop()
                partner_homolog = pair_dict.get(int(partner)) # Partner_homolog is the L-R pair of partner
                
                if partner_homolog is None:
                    # No L-R pair to merge, but must normalize (partner:L/R-pair synapses ÷ sum L/R synapses)
                    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values / sum0
                    column0.append(col0)
                    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values / sum1
                    column1.append(col1)
                else:
                    # Remove from next iteration
                    partner_skids.discard(partner_homolog)
                    # Merge the two vectors (L-R pairs) and normalize ()
                    col0 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])].values) / sum0
                    column0.append(col0)
                    col1 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])].values) / sum1
                    column1.append(col1)

            # Obtain columns of merged/normalized values (initially as ndarray of 1-item ndarrays) and convert into adjacency matrix
            flat_col0 = np.array(list(chain.from_iterable(column0)))
            flat_col1 = np.array(list(chain.from_iterable(column1)))
            matrix = pd.DataFrame([flat_col0, flat_col1])

            # # Calculate connectivity score for pair and append to sims
            score = navis.connectivity_similarity(adjacency = matrix, metric = metric, n_cores= 1)
            sims.append(score.iloc[0][1])
            # print(sims)
            
    # Output scores as pd.DF, int columns for skids & float for scores 
    out = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
    out["cosine_similarity"] = pd.Series(np.asarray(sims, dtype=float))

    return out  

def isolate_pair(pair):
    pair_dict = {}
    for skids in brain_pairs:
        pair_dict[skids[0]] = skids[1]
        pair_dict[skids[1]] = skids[0]

    partners = pymaid.get_partners(list(pair))
    partners = partners[partners.relation == 'upstream']
    print(partners)
    partner_skids = set(partners["skeleton_id"])
    sum0 = float(sum(partners.iloc[:, 4]))
    if sum0 == 0:
        score = np.NaN
    sum1 = float(sum(partners.iloc[:, 5]))
    if sum1 == 0:
        score = np.NaN

    if sum0 and sum1 != 0:

        column0 = []
        column1 = []
        while partner_skids:
            partner = partner_skids.pop()
            partner_homolog = pair_dict.get(int(partner))

            if partner_homolog is None:
                col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] / sum0
                column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
                col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
                column1.append(pd.to_numeric(col1.values, errors = 'coerce'))
            else:
                col0 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])].values) / sum0
                column0.append(pd.to_numeric(col0, errors = 'coerce'))
                col1 = (partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])].values + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])].values) / sum1
                column1.append(pd.to_numeric(col1, errors = 'coerce'))

            partner_skids.discard(partner)
            partner_skids.discard(partner_homolog) 

        column0 = pd.Series(column0).values
        column1 = pd.Series(column1).values
        flat_col0 = np.array(list(chain.from_iterable(column0)))
        flat_col1 = np.array(list(chain.from_iterable(column1)))
        matrix = pd.DataFrame([flat_col0, flat_col1])
        score = navis.connectivity_similarity(adjacency = matrix, metric = 'cosine', n_cores= 1)
        score = score.iloc[0][1]
        print(score)
    return score

def isolate_partner(pair, partner):
    pair_dict = {}
    for skids in brain_pairs:
        pair_dict[skids[0]] = skids[1]
        pair_dict[skids[1]] = skids[0]

    partners = pymaid.get_partners(list(pair))
    partners = partners[partners.relation == 'upstream']
    partner_skids = set(partners["skeleton_id"])
    sum0 = float(sum(partners.iloc[:, 4]))
    sum1 = float(sum(partners.iloc[:, 5]))
    
    if sum0 and sum1 != 0:

        partner_homolog = pair_dict.get(partner)
                        
        if partner_homolog is None:
            col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] / sum0
            col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
        else:
            col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])] / sum0
            col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])] / sum1

    else:
        print('No comparison to be made, owing to a sum of zero synapses in at least one pair')

#Run individual input/output analyses

# mI = merged_input(skid_pairs)
# mO = merged_output(skid_pairs)
# # %%
# input = mI
# input = input.sort_values('left_skid')
# input = input.reset_index(drop = True)

# vals = pymaid.get_names(list(input['left_skid']))
# vals = list(vals.values())
# vals.insert(154, 'AVL011PN (2!)')
# vals.insert(341, 'OLP4;left')
# input["left_name"] = pd.Series(vals)
# input = input.sort_values('cosine_similarity')

# output = mO
# output = output.sort_values('left_skid')
# output = output.reset_index(drop = True)
# output["left_name"] = pd.Series(vals)
# output = output.sort_values('cosine_similarity')


