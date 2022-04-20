#%%
from cmath import nan
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



METRIC = "cosine"

bp = pd.read_csv(HERE / "brain-pairs.csv")
bp.drop('Unnamed: 0', axis=1, inplace=True)
skid_pairs = bp.to_numpy().astype(int).tolist()

def merged_analysis(pair_list, metric = "cosine", is_input = False):

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

        # try:
        sum0 = float(sum(partners.iloc[:, 4]))
        # except sum0 == 0:
        #     sum0 = 0 + 10**-100
        # try:
        sum1 = float(sum(partners.iloc[:, 5]))
        # except sum1 == 0:
        #     sum1 = 0 + 10**-500

        column0 = []
        column1 = []
        while partner_skids:
            partner = partner_skids.pop()
            partner_homolog = pair_dict.get(partner) 
                
            if partner_homolog is None:
                try:
                    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] / sum0
                except ZeroDivisionError:
                    col0 = np.NaN
                column0.append(pd.to_numeric(col0.values, errors = 'coerce'))
                try:
                    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] / sum1
                except ZeroDivisionError:
                    col1 = np.NaN
                column1.append(pd.to_numeric(col1, errors = 'coerce'))
            else:
                    # Fuse two vectors, normalized
                try:
                    col0 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[0])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[0])] / sum0
                except ZeroDivisionError:
                    col0 = np.NaN
                column0.append(pd.to_numeric(col0, errors = 'coerce'))
                try:
                    col1 = partners.loc[partners['skeleton_id'] == str(partner)][str(pair[1])] + partners.loc[partners['skeleton_id'] == str(partner_homolog)][str(pair[1])] / sum1
                except ZeroDivisionError:
                    col1 = np.NaN
                column1.append(pd.to_numeric(col1, errors = 'coerce'))
                
                # Remove both from set
            partner_skids.discard(partner)
            partner_skids.discard(partner_homolog)         
        column0 = pd.Series(column0).values
        column1 = pd.Series(column1).values
        print(column0, column1)

        flat_col0 = []
        for sublist in column0:
            for item in sublist:
                flat_col0.append(item)

        flat_col1 = []
        for sublist in column1:
            for item in sublist:
                flat_col1.append(item)

            
        matrix = pd.DataFrame([flat_col0, flat_col1])
        print(matrix)
        score = navis.connectivity_similarity(adjacency = matrix, metric = metric, n_cores = 1)
        print(score)
        sims.append(score.iloc[0][1])


    print(sims)

merged_analysis(skid_pairs[0:10])


sys.exit()
    # Output scores as pd.DF, int columns for skids & float for scores 
out = pd.DataFrame(pair_list, dtype=int, columns=["left_skid", "right_skid"])
out["cosine_similarity"] = pd.Series(np.asarray(sims, dtype=float))

# %%
