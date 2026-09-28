# PsuprProj

Tools for preparing and inspecting CoughVID metadata and audio files.

## Requirements

- Python 3.10 or newer
- The CoughVID dataset placed in the repository's `data/` directory

The `data/` directory is intentionally ignored by Git because the dataset is
large. The project can download the public archive directly from
[Zenodo record 7024894](https://zenodo.org/records/7024894):

```text
data/coughvid_20211012/metadata_compiled.csv
data/coughvid_20211012/<uuid>.<audio-extension>
data/coughvid_20211012/<uuid>.json
```

## Setup

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd PsuprProj
```

Create a virtual environment. On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download and extract the dataset automatically:

```bash
python -m psupr_proj --download
```

This downloads the 2.3 GB `public_dataset_v3.zip` archive into `data/`,
verifies its Zenodo MD5 checksum, and extracts `data/coughvid_20211012/`.
The download resumes from a partial `.part` file if interrupted. Make sure
there is at least 5 GB of free disk space.

For development and optional `pytest` support:

```bash
python -m pip install -r requirements-dev.txt
```

## Workflow

Summarize the source metadata:

```bash
python -m psupr_proj
```

Copy records with `cough_detected >= 0.8`, including matching audio and JSON
files, into `data/coughvid_20211012_processed/`:

```bash
python -m psupr_proj --filter
```

Analyze the filtered metadata:

```bash
python -m psupr_proj --analyze
```

Clean the filtered metadata for supervised training:

```bash
python -m psupr_proj --clean
```

The cleaning command writes `data/coughvid_20211012_processed/metadata_clean.csv`.

Use a different threshold or output directory when needed:

```bash
python -m psupr_proj --filter --threshold 0.9 --output-dir data/processed
```

The cough score threshold is a preprocessing hyperparameter shared by
filtering and cleaning. For comparable experiments, use separate output
directories so the generated metadata and audio do not overwrite each other:

```bash
# Experiment A: cough_detected >= 0.80
python -m psupr_proj --filter --threshold 0.80 \
	--output-dir data/experiment_a
python -m psupr_proj --clean --threshold 0.80 \
	--output-dir data/experiment_a

# Experiment B: cough_detected >= 0.70
python -m psupr_proj --filter --threshold 0.70 \
	--output-dir data/experiment_b
python -m psupr_proj --clean --threshold 0.70 \
	--output-dir data/experiment_b
```

Both experiments then use the same split, model, and evaluation commands;
only the preprocessing threshold differs.

### Preprocessing experiments

The P0-P4 experiments reuse the same `metadata_split.csv`, CNN, fixed
Log-Mel parameters, and validation evaluator. Only waveform preprocessing
changes:

| Experiment | Trim | Active extraction | Segmentation | Augmentation |
| --- | --- | --- | --- | --- |
| P0 | No | No | No | No |
| P1 | Yes | No | No | No |
| P2 | No | Yes | No | No |
| P3 | No | Yes | Yes | No |
| P4 | No | Yes | Yes | Yes |

Run one experiment from the project directory:

```bash
python -m psupr_proj --train-p0 --output-dir data/experiment_a
python -m psupr_proj --train-p1 --output-dir data/experiment_a
python -m psupr_proj --train-p2 --output-dir data/experiment_a
python -m psupr_proj --train-p3 --output-dir data/experiment_a
python -m psupr_proj --train-p4 --output-dir data/experiment_a
```

Compare primarily COVID-19 recall, COVID-19 F1, macro F1, and the confusion
matrix with:

```bash
python -m psupr_proj --evaluate-model p4_balanced_sampling \
	--output-dir data/experiment_a
```

To train and evaluate all five experiments sequentially, first create one
shared split and then run:

```bash
chmod +x scripts/run_p0_p4.sh
./scripts/run_p0_p4.sh data/experiment_a
```

The script writes one model, training plot, and evaluation report per preset
under the same output directory. Set `PYTHON_BIN` when the environment's
Python executable is not on `PATH`:

```bash
PYTHON_BIN=../.venv/bin/python ./scripts/run_p0_p4.sh data/experiment_a
```

The commands use these defaults:

- Input metadata: `data/coughvid_20211012/metadata_compiled.csv`
- Filtered output: `data/coughvid_20211012_processed/`
- Filtered metadata: `metadata_filtered.csv`
- Clean metadata: `metadata_clean.csv`

## Tests

Run the standard-library test suite:

```bash
python -m unittest discover -s tests
```

## Git Workflow

`main` is the protected integration branch. Teammates should create a feature
branch, push it, and open a pull request into `main`:

```bash
git switch -c my-feature
git add .
git commit -m "Describe the change"
git push -u origin my-feature
```

The pull request must pass the `Python checks / syntax` workflow, which runs
Python compilation and the test suite. Direct pushes to `main` should be
disabled in the GitHub branch protection settings.

## Project Layout

```text
src/psupr_proj/    Package source
tests/             Automated tests
data/              Local dataset, ignored by Git
pyproject.toml     Package metadata and build configuration
requirements.txt   Runtime dependency file
requirements-dev.txt Development dependency file
```
