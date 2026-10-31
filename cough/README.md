# PsuprProj

Tools for preparing and inspecting CoughVID metadata and audio files.

## Requirements

- Python 3.10 or newer
- The CoughVID dataset placed in the repository's `data/` directory

The `data/` directory is intentionally ignored by Git because the dataset is large. The public archive is available from https://zenodo.org/records/7024894.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m psupr_proj --download
```

## COUGHVID workflow

```bash
python -m psupr_proj
python -m psupr_proj --filter
python -m psupr_proj --analyze
python -m psupr_proj --clean
```

## P0-P4 preprocessing experiments

All experiments use a shared dataset split, CNN architecture and Log-Mel configuration.

| Experiment | Trim | Active extraction | Segmentation | Augmentation |
|---|---|---|---|---|
| P0 | No | No | No | No |
| P1 | Yes | No | No | No |
| P2 | No | Yes | No | No |
| P3 | No | Yes | Yes | No |
| P4 | No | Yes | Yes | Yes |

```bash
python -m psupr_proj --train-p0 --output-dir data/experiment_a
python -m psupr_proj --train-p1 --output-dir data/experiment_a
python -m psupr_proj --train-p2 --output-dir data/experiment_a
python -m psupr_proj --train-p3 --output-dir data/experiment_a
python -m psupr_proj --train-p4 --output-dir data/experiment_a
```

Run all presets:

```bash
./scripts/run_p0_p4.sh data/experiment_a
```

Evaluate:

```bash
python -m psupr_proj --evaluate-model p4_balanced_sampling --output-dir data/experiment_a
```

## Tests

```bash
python -m unittest discover -s tests
```

## Coswara and HeAR

The project also includes Coswara baseline, imbalance and HeAR experiments. The source dataset is available at https://github.com/iiscleap/Coswara-Data.

See `scripts/run_coswara_baselines.py`, `scripts/run_coswara_imbalance_experiments.py` and `src/psupr_proj/coswara_hear_stage2.py`.

Experiment results should be archived separately from raw audio datasets.
