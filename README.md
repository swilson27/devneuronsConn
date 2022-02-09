# CAnalysis

Systematic connectivity analysis across L-R pairs of brain; conducted for 'deviant neurons' project. Code and small data found within.

# Usage

- Ensure python 3.8+ is available
- create and activate a virtual environment with your tool of choice
  - e.g. `python3 -m venv --prompt CAnalysis venv/ && source venv/bin/activate`
- install requirements: `pip install -r requirements.txt`
- Copy `.env.example` (e.g. `cp .env.example .env`) and fill in your own CATMAID credentials
  - This **must** not be version controlled. Files called `.env` are not tracked by git (because it's listed in `.gitignore`)
  - Load the credentials into your environment with `source .env`. Some tools, like dotenv and direnv, can do this automatically.
