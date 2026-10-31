# Respiratory Audio Experiments — COUGHVID, Coswara and HeAR

This README documents the respiratory-audio workstream of the multimodal COVID-19 classification project. It covers COUGHVID preparation and P0–P4 experiments, Coswara baseline and imbalance experiments, and exploratory Google HeAR feature extraction. These are research experiments, not a clinically validated diagnostic system.

**Working directory:** commands in this README are run from `PsuprProj/cough/` unless stated otherwise. The repository root is `PsuprProj/`.

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
git clone https://github.com/wtwenting/multimodalLungDiseaseClassifier.git PsuprProj
cd PsuprProj/cough
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

## Reproducibility: source scripts, results and report mapping

The mapping below is based on the project file inventory and documented workflow. **File existence is confirmed from the local inventory, but the exact generating CLI for Coswara and HeAR has not been independently verified from the Python argument parsers.** Do not assume that running a script without options reproduces the recorded results.

All script and result paths in the table are relative to `cough/`. The report locations refer to Chapter 3 of the group report.

| Report section | Experiment / task | Source script or entry point | Observed local result |
| --- | --- | --- | --- |
| 3.1.1 | COUGHVID download, filter, analysis and cleaning | `python -m psupr_proj` (see Workflow above) | `data/experiment_b/metadata_clean.csv` |
| 3.1.1 | COUGHVID P0–P4 preprocessing comparison | `scripts/run_p0_p4.sh` / `python -m psupr_proj --train-p0` … `--train-p4` | `data/experiment_b/p0_p4_summary.json`, `p0_p4_summary.csv` |
| 3.1.1 | COUGHVID P0–P4 evaluation | `python -m psupr_proj --evaluate-model` | `data/experiment_b/models/p0_balanced_sampling.evaluation.json` … `p4_balanced_sampling.evaluation.json` |
| 3.1.1 | COUGHVID HeAR H0–H2 | `src/psupr_proj/run_coughvid_hear_h0_h2.py`; `run_coughvid_hear_overnight.sh` | **Result location to verify** |
| 3.1.1 | COUGHVID HeAR H3 | `src/psupr_proj/run_coughvid_hear_h3.py`; `run_coughvid_hear_h0_h3_overnight.sh` | **Result location to verify** |
| 3.1.1 | HeAR embeddings and dense downstream model | `src/psupr_proj/hear_features.py`; `src/psupr_proj/train_hear_dense.py` | `artifacts/hear_dense/metrics.json`, `history.json` |
| 3.1.1 | HeAR Logistic Regression and threshold experiments | `src/psupr_proj/train_hear_logreg.py`; `src/psupr_proj/tune_hear_threshold.py` | **Result location to verify** |
| 3.1.2 | Coswara dataset preparation | `src/psupr_proj/data_coswara.py` | **Generated dataset manifests to verify** |
| 3.1.2 | Coswara heavy-cough EDA and spectral energy | `src/psupr_proj/coswara_spectral_energy_eda.py` | `data/coswara/stage1EDA/coswara_heavy_valid_eda_summary.json`, `coswara_heavy_valid_spectral_energy_summary.json` |
| 3.1.2 | Coswara 1D CNN, 2D CNN and LSTM baselines | `scripts/run_coswara_baselines.py` | `data/coswara/baseline_runs/coswara_baseline_summary.json` and `models/*.results.json` |
| 3.1.2 | Coswara B1–B3 imbalance experiments | `scripts/run_coswara_imbalance_experiments.py` | `data/coswara/imbalance_runs/imbalance_experiments_summary.json` and per-configuration `results.json` |
| 3.1.2 | Coswara handcrafted features | `src/psupr_proj/coswara_stage2_features.py` | `data/coswara/stage2/handExtract/coswara_feature_summary.csv` |
| 3.1.2 | Coswara exploratory HeAR features/classification | `src/psupr_proj/coswara_hear_stage2.py` | `data/coswara/stage2/hear_test_run/hear_exploratory_metrics.json`, `hear_feature_summary.csv` |

### Coswara dataset provenance

The original Coswara dataset is maintained by the IISc LEAP group at <https://github.com/iiscleap/Coswara-Data>. Dataset download and organization are **external to the COUGHVID `--download` command**. Obtain and prepare Coswara separately in accordance with its repository instructions and licence. The local experiment directory observed in the inventory is `cough/data/coswara/`; its presence does not imply that raw audio is version-controlled.

### Reproducing experiments from the CLI

**Verified documented COUGHVID workflow** (from the `cough/` directory):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m psupr_proj --download
python -m psupr_proj --filter
python -m psupr_proj --analyze
python -m psupr_proj --clean
python -m psupr_proj --train-p0 --output-dir data/experiment_a
python -m psupr_proj --evaluate-model p0_balanced_sampling --output-dir data/experiment_a
PYTHONPATH=src python -m unittest discover -s tests -v
```

The existing **Preprocessing experiments** section above provides the complete P0–P4 CLI sequence. For Coswara and HeAR, inspect the actual script help/argument parser and the run shell scripts before reproducing; exact command-line flags and output locations have not yet been validated here. Example inspection commands (no training started):

```bash
python scripts/run_coswara_baselines.py --help
python scripts/run_coswara_imbalance_experiments.py --help
sed -n '1,160p' run_coughvid_hear_h0_h3_overnight.sh
```

Some scripts may not implement `--help`; inspect their `argparse` definitions if these commands do not work.

### Archiving and privacy

Keep original audio, per-recording JSON metadata, participant-level splits, and any patient-linked predictions out of public Git unless explicitly permitted. For academic review, archive only vetted aggregate metrics, configurations, and experiment summaries in `cough/experiment_records/`. The listed results above are **local paths**, not confirmation that all files have already been committed.

Suggested archive categories: `coughvid/`, `hear/`, `coswara_baseline/`, `coswara_imbalance/`, `coswara_eda/`, `coswara_features/`, and `coswara_hear/`. Record the Git commit used to generate each result.

### Environment and limitations

`requirements.txt` specifies minimum runtime versions, not an exact lock. `requirements-dev.txt` adds pytest. Record the actual Python and library versions used for each final experiment before claiming bit-for-bit reproducibility. A saved summary and a source script are evidence of an experiment, but exact reproduction also requires the dataset version, split, seeds, pretrained weights, and original configuration.

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
