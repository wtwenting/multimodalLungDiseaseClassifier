# BIMCV paired X-ray + CT fusion experiments

This README documents the original demonstration baseline and the later frozen-feature CT–X-ray fusion experiments. **These are different experimental pipelines**: do not treat results from the initial from-scratch baseline as directly comparable with results from the later patient-matched frozen-encoder pipeline.

## 1. Original demonstration baseline

The original README instructed users to place `train_fusion.py` at `ensemble/train_fusion.py` and `requirements.txt` at `ensemble/requirements.txt`. Verify these files exist in your checkout before running the commands below.

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r ensemble/requirements.txt
python ensemble/train_fusion.py --mode smoke
python ensemble/train_fusion.py --mode train --model xray --epochs 10
python ensemble/train_fusion.py --mode train --model ct --epochs 10
python ensemble/train_fusion.py --mode train --model fusion --epochs 10
```

Input: `dataAnalysis/output/paired_patient_manifest.csv`, containing `patient_id`, `covid_status`, `xray_path`, and `ct_path`.

The smoke test checks forward and backward passes using two local paired patients. Training writes patient splits, checkpoints and evaluation metrics to `ensemble/output/`. Splits are saved and reused for all three baseline configurations. If the eligible cohort changes, `--rebuild-splits` starts a **new experiment**; all comparison models must then be rerun using the new split.

This is a small, from-scratch demonstration baseline, **not** the selected final frozen-feature classifier. The `--ct-intensity percentile` option is appropriate when stored intensity units are unknown; use `--ct-intensity hu` only if Hounsfield units are confirmed. `--ct-shape D H W` controls 3D input dimensions. Patient matching alone does not establish same-session imaging. BIMCV positive/negative labels should not be equated with healthy/ill.

## 2. Later controlled frozen-feature fusion

The selected later-phase pipeline uses a frozen CT Encoder B representation (1,024 features per patient) and a combined frozen X-ray Model B + Model C representation (256 features per patient). CT features precede X-ray features in the final 1,280-dimensional vector. The downstream selected model is a `StandardScaler` followed by L2-regularized Logistic Regression (`C=0.1`, `liblinear`, no class weighting, decision threshold 0.5).

The controlled downstream comparisons use **446 training patients** and **97 validation patients**. The recorded final evaluation uses **99 test patients**. The LR, MLP and separate-PCA experiments share the later-phase outer train/validation patient cohort, but do not necessarily use identical internal cross-validation folds. Earlier exploratory fusion experiments used different configurations and patient partitions.

The scripts below generally expect execution **from the `ensemble/` directory**, using existing frozen feature caches. These caches and original medical images are not included in the lightweight Git experiment-record archive.

```bash
cd ensemble
# Example: run an alignment-only check before training
python src/run_fusion_baselines_overnight.py --dry-run
```

## 3. Numbered experiment, script and result mapping

Numbering is independent of report chapter numbers. All paths in this table are relative to `ensemble/`. The `experiment_records/` column refers to selected **archived aggregate results**; the `output/` column refers to original local run outputs. A file's presence in the archive does not by itself establish that it can be reproduced without the upstream feature caches, checkpoints and datasets.

| No. | Experiment / report evidence | Script | Local result(s) | Archived result(s) |
| --- | --- | --- | --- | --- |
| 1 | Initial paired CT/X-ray demonstration baseline | `train_fusion.py` (original baseline; verify location) | `output/` (original baseline outputs; exact filenames depend on run) | Not separately archived |
| 2 | Early exploratory frozen-feature fusion | `src/train_mixed_fusion.py` (verify tracked version) | `output/mixed_fusion/experiments/fusion/metrics.json`; `output/mixed_fusion/frozen_mixed_fusion_metrics.json` | `experiment_records/exploratory_fusion/metrics.json`; `experiment_records/early_frozen_fusion/frozen_mixed_fusion_metrics.json` |
| 3 | CT encoder local verification | `src/test_ct_encoder.py` (verify exact producing command) | `output/ct_v61_frozen_baseline/summary.json` | `experiment_records/ct_encoder_verification/summary.json` |
| 4 | CT eligibility / quality audit | Producing script **not yet confirmed** | `output/ct_v5_encoder_audit/ct_v5_encoder_eligibility_summary.json` | `experiment_records/ct_eligibility/ct_v5_encoder_eligibility_summary.json` |
| 5 | X-ray preprocessing comparison | `src/benchmark_xray_preprocessing.py` (verify exact run) | `output/xray_preprocessing_benchmark/benchmark_summary.json` | `experiment_records/xray_preprocessing/benchmark_summary.json` |
| 6 | X-ray B, C and B+C comparison | `src/compare_encoders.py` (verify exact run) | `output/xray_dual_encoder_256d/experiment_summary.json` | `experiment_records/xray_encoder_comparison/experiment_summary.json` |
| 7 | X-ray patient aggregation | `src/benchmark_patient_aggregation.py` | `output/xray_patient_feature_aggregation/experiment_summary.json` | `experiment_records/xray_patient_aggregation/experiment_summary.json` |
| 8 | X-ray random-seed checks | Producing script **not yet confirmed** | `output/xray_final_checks/seed_results.csv`; `seed_summary.csv` | `experiment_records/xray_seed_checks/seed_results.csv`; `seed_summary.csv` |
| 9 | Controlled CT-only, X-ray-only and CT+X-ray LR | `src/run_fusion_baselines_overnight.py` | `output/fusion_lr_overnight/summary.json` | `experiment_records/baseline_lr/summary.json` |
| 10 | MLP comparison | `src/run_fusion_mlp.py` | `output/fusion_mlp/summary.json` | `experiment_records/mlp/summary.json` |
| 11 | Separate CT/X-ray PCA comparison | `src/run_fusion_separate_pca.py` | `output/fusion_separate_pca/summary.json` | `experiment_records/separate_pca/summary.json` |
| 12 | Paired bootstrap uncertainty (5,000 iterations; seed 42) | `src/audit_fusion_improvement.py` | `output/fusion_paired_bootstrap/paired_bootstrap_summary.json` | `experiment_records/bootstrap/paired_bootstrap_summary.json` |
| 13 | Modality-block permutation (500 iterations; seed 42) | `src/audit_fusion_modality_ablation.py` | `output/fusion_modality_ablation/modality_ablation_summary.json` | `experiment_records/permutation/modality_ablation_summary.json` |
| 14 | Freeze and reproduce selected LR | `src/freeze_ct_xray_fusion.py` | `output/fusion_frozen_final/freeze_report.json` | `experiment_records/final_frozen_model/freeze_report.json` |
| 15 | Final 99-patient test evaluation | `src/evaluate_fusion_test.py` | `output/fusion_final_test/final_test_summary.json`; `preflight.json` | `experiment_records/final_test/final_test_summary.json`; `preflight.json` |

The mappings marked **verify** are based on file names and the recorded workflow, not an independently confirmed end-to-end provenance audit. Confirm the generating command and Git commit before claiming exact reproducibility for those rows.

## 4. Command-line execution guide (CLI)

**Working directory:** Run the later frozen-feature commands from `ensemble/` (not the repository root). The commands below reproduce the *analysis steps* only if the original feature caches, prediction files, dependencies and appropriate model artifacts are available. They do not automatically download BIMCV data or recreate the upstream encoders.

```bash
cd ~/Projects/PsuprProj/ensemble
# Activate the environment used for the original runs; adjust the path if needed.
source ../.venv/bin/activate
python --version
python -m pip show numpy scikit-learn joblib
```

### 4.1 Verify inputs without training

```bash
python src/run_fusion_baselines_overnight.py --dry-run
python src/run_fusion_mlp.py --dry-run
python src/run_fusion_separate_pca.py --dry-run
```

These dry runs check available training/validation feature indexes and patient matching; they do not fit downstream models. Expected feature inputs include:

- `output/ct_v61_frozen_baseline/B_mean_features_1024d.npy` and `B_mean_feature_index.csv`
- `output/ct_v61_frozen_baseline/A_topk_features_1024d.npy` and `A_topk_feature_index.csv` (baseline comparison)
- `output/xray_patient_feature_aggregation/patient_features_256d.npy` and `patient_aggregation_index.csv`

The arrays and patient-level indexes are not part of the lightweight Git archive. The same outer 446-training/97-validation cohort is used by the later experiments; their internal CV folds differ.

### 4.2 Run downstream comparisons

```bash
# Controlled CT A/B, X-ray, and CT+X-ray Logistic Regression baselines (5-fold CV default)
python src/run_fusion_baselines_overnight.py

# Frozen CT B + X-ray MLP comparison (3-fold CV default; seeds 42, 123, 2026)
python src/run_fusion_mlp.py

# Separate PCA for CT and X-ray, then LR/MLP (3-fold CV default)
python src/run_fusion_separate_pca.py
```

**Caution:** These commands write to the default `output/` directories and may overwrite existing summary files. To preserve the original evidence, supply a new output directory, for example:

```bash
python src/run_fusion_mlp.py --out output/reproduction_fusion_mlp
```

### 4.3 Run uncertainty and modality-dependence audits

These audits use saved predictions or feature caches from the controlled baseline run.

```bash
# 5,000 stratified paired patient bootstrap samples; seed 42
python src/audit_fusion_improvement.py --iterations 5000 --seed 42 \
  --out output/reproduction_paired_bootstrap

# 500 validation-set block permutations per modality; seed 42
python src/audit_fusion_modality_ablation.py --iterations 500 --seed 42 \
  --out output/reproduction_modality_ablation
```

Bootstrap requires `output/fusion_lr_overnight/{CT_B_only,Xray_only,CT_B_plus_Xray}_validation_predictions.csv`. Permutation additionally requires `output/fusion_lr_overnight/summary.json` and the CT/X-ray frozen feature caches. It refits the *already selected* LR configuration once on training data, verifies baseline validation metrics, and keeps the model fixed while permuting features. Neither audit reads test rows.

### 4.4 Freeze the selected classifier

```bash
python src/freeze_ct_xray_fusion.py --out output/reproduction_frozen_final
```

This command expects the saved `CT_B_plus_Xray_validation_predictions.csv` from the original controlled baseline. It checks exactly 446 training and 97 validation patients, reproduces the saved validation probabilities within the configured tolerance, and then writes the frozen model and `freeze_report.json`. **Do not replace the original frozen model used for final testing without documenting a new run.**

### 4.5 Upstream feature extraction and final test

The following scripts are tracked, but their complete command lines depend on local checkpoints, data manifests and input paths. Consult `--help` and their source code before running; the exact original commands are **not verified** here:

```bash
python src/test_ct_encoder.py --help
python src/benchmark_patient_aggregation.py --help
python src/extract_ct_test_features.py --help
python src/extract_xray_test_features_reviewed.py --help
python src/evaluate_fusion_test.py --help
```

**Do not run the final test evaluation as part of routine model selection.** The 99-patient result in the archive is the recorded final frozen-model evaluation, not a new independent test. If a script does not implement `--help`, inspect its arguments in the source rather than assuming a command will work.

### 4.6 Check result files

```bash
find experiment_records -type f | sort
find output/fusion_lr_overnight output/fusion_mlp output/fusion_separate_pca \
  -maxdepth 1 -name 'summary.json' -print
```

The `experiment_records/` folder is the reviewed evidence archive. `output/` contains local run products and may also contain exploratory or repeated experiments. Avoid copying results from a new run over archived originals.

## 5. Tracked scripts and input dependencies

Previously confirmed as tracked in Git:

- Downstream selection: `src/run_fusion_baselines_overnight.py`, `src/run_fusion_mlp.py`, `src/run_fusion_separate_pca.py`.
- Audits: `src/audit_fusion_dataset.py`, `src/audit_fusion_improvement.py`, `src/audit_fusion_modality_ablation.py`.
- Finalization: `src/freeze_ct_xray_fusion.py`, `src/evaluate_fusion_test.py`.
- Upstream/test utilities: `src/pretrained_encoders.py`, `src/benchmark_patient_aggregation.py`, `src/benchmark_xray_preprocessing.py`, `src/test_ct_encoder.py`, `src/extract_ct_test_features.py`, `src/extract_xray_test_features.py`, `src/extract_xray_test_features_reviewed.py`.

The later downstream scripts require previously generated feature arrays and matching CSV indexes, typically under:

- `output/ct_v61_frozen_baseline/` — `B_mean_features_1024d.npy` and `B_mean_feature_index.csv`.
- `output/xray_patient_feature_aggregation/` — `patient_features_256d.npy` and `patient_aggregation_index.csv`.

The baseline runner also uses CT A (`A_topk_features_1024d.npy` and index). The exact scripts used to create **all** training/validation feature caches should be checked against the final repository. Final test feature extraction is handled separately by the tracked test extraction scripts.

To check tracking in your checkout:

```bash
git ls-files ensemble/src/ | grep -E 'run_fusion|audit_fusion|freeze_ct_xray|evaluate_fusion_test|extract_.*test|benchmark_patient_aggregation'
git ls-files ensemble/experiment_records/
```

## 6. Interpreting the results

The later controlled validation comparison reported CT-only Macro F1 / ROC-AUC of **0.7526 / 0.7770**, X-ray-only **0.7088 / 0.7496**, and CT+X-ray fusion **0.7726 / 0.8040**. The recorded final fusion test evaluation reported **0.7475 Macro F1** and **0.8021 ROC-AUC** on 99 matched patients. Refer to the archived JSON summaries for full metrics and configurations rather than copying rounded numbers into other analyses.

Paired bootstrap intervals for fusion improvements over the unimodal baselines include zero; therefore the validation advantage is **not statistically conclusive**. The permutation analysis measures dependence on feature groups, **not** causal or clinical importance. Earlier exploratory experiments accessed different partitions, so the final test cohort must not be described as untouched throughout the entire project history.

## 7. Archiving and privacy

The `experiment_records/` directory contains selected aggregate summaries, not the complete raw experiment outputs. Preserve original output files in approved restricted storage. Do **not** commit raw medical images, patient-level prediction tables, patient split manifests, feature matrices, model checkpoints, secrets or identifiable local image paths without appropriate permissions and review. Archived copies may redact patient-specific paths; retain original files separately for provenance.

Before committing:

```bash
git status --short
grep -RniE 'patient_id|/Users/|api_key|password|secret|token' ensemble/experiment_records/
git add ensemble/README.md ensemble/experiment_records/
git diff --cached --name-only
```

Inspect all staged files; the grep check alone cannot guarantee privacy. If your README is stored elsewhere, adjust its path in the `git add` command. Do not use `git add .` to stage the entire working tree unintentionally.

## 8. Reproducibility limits

The repository and aggregate summaries provide a traceable account of experiment configuration and outcomes, but a full rerun additionally requires the original permitted BIMCV data, the correct frozen encoder checkpoints, compatible preprocessing, exact feature caches or their generating scripts, dependency versions, and patient split manifests in approved storage. Same-patient matching does not prove that CT and X-ray examinations were acquired during the same clinical session. This system is a research demonstration, **not a clinically validated diagnostic tool**.

## Appendix reference

See Appendix C for the numbered source-code and archived-result inventory. The numbering is independent of the report chapter structure.
