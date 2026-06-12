Process Connectome Mapper 3 output


# Setup
1. install `uv`
2. `uv sync`


# Run

See the usage document
```bash
uv run code/connectome_analysis.py --help
```

Run for all types of weights
```bash
# activate uv venv
source .venv/bin/activate
bash run_all_weights.sh <SUBJECT_DIR>
```