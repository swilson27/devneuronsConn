#%%
import numpy as np
import pandas as pd
from pathlib import Path
import pymaid
import json
from matplotlib import pyplot as plt
import os

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)



### Load data ###



#%%
with open(OUT_DIR / 'merged_cosine_th1_input.json') as json_file:
    input = json.load(json_file)
with open(OUT_DIR / 'merged_cosine_th1_output.json') as json_file:
    output = json.load(json_file)
with open(OUT_DIR / 'merged_cosine_th2_input.json') as json_file:
    input2 = json.load(json_file)
with open(OUT_DIR / 'merged_cosine_th2_output.json') as json_file:
    output2 = json.load(json_file)

#%%
canalysis_input = pd.read_csv(HERE / 'canalysis_cosine_th2_input.csv')
canalysis_output = pd.read_csv(HERE / 'canalysis_cosine_th2_output.csv')

#%%
'''
input_vals = {key:val for key, val in input.items() if val != None}
sorted_IV=[] # IV means input values
# w is connectivity metric
for w in sorted(input_vals, key=input_vals.get, reverse = False):
                sorted_IV.append([w, input[w]])

output_vals = {key:val for key, val in output.items() if val != None}
sorted_OV=[] # Output values
for w in sorted(output_vals, key=output_vals.get, reverse = False):
                sorted_OV.append([w, output[w]])

#%%
with open("sorted_conn_inputs.json", 'w') as outfile:
    json.dump(sorted_IV, outfile)

with open("sorted_conn_outputs.json", 'w') as outfile:
    json.dump(sorted_OV, outfile)
'''



### Explore data ###



#%%
def gen_xy(data, normalise = False):   
    """ Generate x and y values for plotting from connectivity analysis data, comparing similarity across L-R pairs
        cosine similarity values on x and # of L-R pairs on y

    Args:
        data: either json nested dict, with connectivity values stored within 'cosine similarity') 
              or pandas df, with connectivity values stored in column 'cosine_similarity'
        normalise (bool): option to normalise y (cumulative sum) values, if plotting for different amounts of values (x). Defaults to False.
    """     
    if type(data) == dict:
        vals = data.get('cosine_similarity', {}) 
        vals = list(vals.values())
        vals = [np.nan if x is None else x for x in vals]
        x = sorted(v for v in vals if not np.isnan(v))
        y = np.arange(len(x))
        if normalise:
            y /= len(x)

    if type(data) == pd.DataFrame:
        vals = list(data['cosine_similarity'] )
        vals = [np.nan if x is None else x for x in vals]
        x = sorted(v for v in vals if not np.isnan(v))
        y = np.arange(len(x))
        if normalise:
            y /= len(x)

    else:
        print('data not of correct format (json dict or pandas df)')

        
    return x, y


## Plots for whole data ##


#%%
fig = plt.figure()
ax = fig.add_subplot()

out_x, out_y = gen_xy(output, False)
ax.plot(out_x, out_y, label="output similarity")

in_x, in_y = gen_xy(input, False)
ax.plot(in_x, in_y, label="input similarity")

ax.legend()
ax.set_xlabel("Cosine similarity value")
ax.set_label("Cumulative frequency")

fig.savefig("canalysis_cumulative_th1.pdf", format="pdf")

#%%
fig2 = plt.figure()
ax = fig2.add_subplot()

out2_x, out2_y = gen_xy(output2, False)
ax.plot(out2_x, out2_y, label="output similarity")

in2_x, in2_y = gen_xy(input2, False)
ax.plot(in2_x, in2_y, label="input similarity")

ax.legend()
ax.set_xlabel("Cosine similarity value")
ax.set_label("Cumulative frequency")

fig2.savefig("canalysis_cumulative_th2.pdf", format="pdf")

#%%
fig3 = plt.figure()
ax = fig3.add_subplot()

out_x, out_y = gen_xy(output, False)
ax.plot(out_x, out_y, label="output, th=1")

in_x, in_y = gen_xy(input, False)
ax.plot(in_x, in_y, label="input, th=1")

out2_x, out2_y = gen_xy(output2, False)
ax.plot(out2_x, out2_y, label="output, th=2")

in2_x, in2_y = gen_xy(input2, False)
ax.plot(in2_x, in2_y, label="input, th=2")

ax.legend()
ax.set_xlabel("Cosine similarity value")
ax.set_label("Cumulative frequency")

fig3.savefig("canalysis_cumulative_th1+2.pdf", format="pdf")


## Lineage analysis ##


#%%
bp = pd.read_csv(HERE / "brain-pairs.csv")
bp.drop('region', axis=1, inplace=True)
brain_pairs = bp.to_numpy().astype(int).tolist()

pair_dict = {}
for skids in brain_pairs:
    pair_dict[skids[0]] = skids[1]
    pair_dict[skids[1]] = skids[0]

#%%

def lineage_subsetter(inputs = True):
    # Iterate through all json files (containing skids of each lineage) in directory, subset inputs or outputs results table and export as CSV
    directory = os.fsencode(HERE / 'lineages')
  
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".json"): 
            
            lineage = filename.replace('.json', '')
            # Extract all skids in each lineage from CATMAID downloaded JSONs
            neurons = pd.read_json(HERE / f"lineages/{filename}")
            neurons = pymaid.get_names(neurons.loc[:]['skeleton_id'])
            # neurons contains both left and right neurons within lineage annotation, 
            # as CATMAID pairs may not completely correspond to those in brain_pairs (and thus canalysis_{input/output} csv)

            # Convert neurons from lineage into integer list of skids
            skids = [int(i) for i in list(neurons.keys())]

            # Subset inputs or outputs CSV with all scores by skids in lineage
            if inputs == True:
                lineage_df = canalysis_input[(canalysis_input['left_skid'].isin(skids)) | (canalysis_input['right_skid'].isin(skids))]
            else:
                lineage_df = canalysis_output[(canalysis_output['left_skid'].isin(skids)) | (canalysis_output['right_skid'].isin(skids))]
               
            # Output to CSV
            lineage_df.to_csv(OUT_DIR / "lineage_csvs"/f"canalysis2_{'in' if inputs else 'out'}put_{lineage}.csv", index=False)

lineage_subsetter(inputs = True)
lineage_subsetter(inputs = False)


## Plots for lineage subsets ##


#%%
fig = plt.figure()
ax = fig.add_subplot()

out_x, out_y = gen_xy(output, False)
ax.plot(out_x, out_y, label="output similarity")

in_x, in_y = gen_xy(input, False)
ax.plot(in_x, in_y, label="input similarity")

ax.legend()
ax.set_xlabel("Cosine similarity value")
ax.set_label("Cumulative frequency")

fig.savefig("canalysis_cumulative_th1.pdf", format="pdf")

