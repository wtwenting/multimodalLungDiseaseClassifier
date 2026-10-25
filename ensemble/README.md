# BIMCV paired X-ray + CT fusion baseline

Copy `train_fusion.py` into `<project>/ensemble/train_fusion.py` and `requirements.txt` into `<project>/ensemble/requirements.txt`.

From the repository root on your Mac:

```bash
source .venv/bin/activate
python -m pip install -r ensemble/requirements.txt
python ensemble/train_fusion.py --mode smoke
python ensemble/train_fusion.py --mode train --model xray --epochs 10
python ensemble/train_fusion.py --mode train --model ct --epochs 10
python ensemble/train_fusion.py --mode train --model fusion --epochs 10
```

Input: `dataAnalysis/output/paired_patient_manifest.csv` (patient_id, covid_status, xray_path, ct_path).

The first smoke test checks forward/backward pass using two local paired patients. Training writes patient splits, best checkpoints and evaluation metrics to `ensemble/output/`. Splits are saved and shared across all three models. If new images add eligible patients, the script refuses to silently change the split: use `--rebuild-splits` to start a **new experiment**, then rerun all baselines and fusion for a fair comparison.

The model is a small from-scratch demonstration baseline, not a pretrained clinical classifier. CT `--ct-intensity percentile` is a robust default when stored intensity units are unknown. Use `--ct-intensity hu` only after confirming Hounsfield units. `--ct-shape D H W` controls 3D input size. All selected patient pairs are based on the existing manifest, which has not validated exam dates or chosen clinically optimal CT series. Labels refer to BIMCV positive/negative cohorts, **not** healthy/ill.

Do not use these models for clinical decisions. For a final study, freeze the test cohort, validate labels and CT preprocessing, tune only on validation data, and consider stronger pretrained encoders.
