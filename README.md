# devneuronsConn


This repository contains scripts and files for a systematic quantification of connectivity similarity across homologous (left:right) pairs of neurons. As a result, any 'deviant' (hence, dev) neurons - which significantly differ from expected alikeness of synaptically connected partners - can be identified from low scores.

Utilises data stored and generated within CATMAID, interfaced with via the Python libraries NAVis and pymaid. Connectivity similarity is algorithmically calculated based on the cosine similarity metric (https://navis.readthedocs.io/en/latest/source/generated/navis.connectivity_similarity.html?highlight=cosine) ; this is applied separately to generate respective scores for each pairs input (i.e. dendritic) and output (i.e. axonic) partners. Files and scripts all utilise connectomic data from 'Seymour', an L1 D. melangoaster larvae.

Links: CATMAID (https://catmaid.readthedocs.io/en/stable/) ; NAVis (https://navis.readthedocs.io/en/latest/) ; pymaid (https://pymaid.readthedocs.io/en/latest/)

Also see devneuronsMorph repo


## Files and scripts


cns-pairs.csv - list of 1640 pairs (left and right) of homologous neurons, represented as CATMAID skeleton IDs (skids). Some are annotated with CNS region. 

script_conn.py - takes each pair and algorithmically generates a cosine similarity score for the input and output fractions of the connectivity matrix

explore_conn.py - script to perform exploratory analysis on data; generates various plots and can be modified for bespoke analyses 

cache and output folders to locally store respective components after running script.


## General information


CATMAID requires credentials, which must be provided by user from a locally stored directory. These will not be tracked by repository's git.

Live CATMAID instances can require lots of data to be downloaded, which both slows scripts and can lead to different results on subsequent runs (if neurons are modified). Intermediate data downloaded from CATMAID locally will be stored within cache/ directory; git will ignore these files.

Output CSVs and exploratory analyses will be stored in the output/ directory to decrease clutter. This is also currently git ignored, but you can modify any of these if desired.


## TO DO: modify below


First use
# Pick a name for your project
PROJECT_NAME="my_project"

# Clone this , then change directory into it
git clone https://github.com/navis-org/pymaid_template.git "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Delete the template's git history and license
rm -rf .git/ LICENSE
# Initialise a new git repo
git init
# Commit the existing files so you can track your changes
git add .
git commit -m "Template from navis-org/pymaid_template"

# Ensuring that you are using a modern version of python (3.9, here), create and activate a virtual environment
python3.9 -m venv --prompt "$PROJECT_NAME" venv
source venv/bin/activate
# use `deactivate` to deactivate the environment

# Install the dependencies
pip install -r requirements.txt

# Make your own credentials file
cp credentials/example.json credentials/credentials.json
Then edit the credentials file to access your desired CATMAID server, instance, and user.


## General use


Whenever you have a new terminal, activate the environment: source venv/bin/activate
Run the script with python script.py
General guidelines
Use a modern python. Many actively maintained scientific tools follow numpy's deprecation schedule: https://numpy.org/neps/nep-0029-deprecation_policy.html
Follow coding standards to make your code as legible and recogniseable as possible. See PEP8: https://www.python.org/dev/peps/pep-0008/
Coding standards sound like nitpicking but they really, really help. e.g. "I know I wrote a function to get data, but was it called getData, GetData, get_data, GET_DATA, or what?". If code is PEP8-compliant, there is only one answer
Auto-formatters (e.g. black, isort) are great for legibility and consistency: use make format to format this repository.
Linters (e.g. flake8) can detect a number of bugs, possible coding issues, and questionable formatting: use make lint to lint this repository (format first).
Documentation makes your code much easier to understand for the next person to read it: that person will probably be you, so it's worthwhile.
Type hints, especially on functions, are also great: https://realpython.com/python-type-checking/#hello-types
Docstrings at the top of modules and functions are better than comments, as they are accessible by the help() function in a python REPL
Use seeded random number generators where randomness is needed, for replicability.
Remember to use ISO-8601 dates/times (YYYY-MM-DD) where necessary.


## License


This repository was based on a template at https://github.com/navis-org/pymaid_template .
