# CT + Chest X-ray Fusion — GUI Integration Handover

**Status:** Frozen research model; inference integration required.
**Date:** 24 October 2026
**Use:** Academic proof of concept, **not a clinical diagnostic tool**.

## 1. Model contract

For one patient, produce a **1,280-dimensional** feature vector:

1. CT: preprocessed 3D NIfTI volume → frozen **V6.1 2.5D DenseNet121 Encoder B**, mean pooling across 12 three-slice views → **1,024D**.
2. X-ray: one or more usable frontal chest radiographs → frozen **Model B v2** (128D) and **Model C ResNet50** (128D) → concatenate **B first, C second** to 256D **per image** → arithmetic mean across retained images of the same patient → **256D**.
3. Concatenate **CT first, X-ray second** → 1,280D. Feed directly to the saved **StandardScaler + LogisticRegression** pipeline; **do not fit a new scaler**.
4. Output positive-class probability and class using **threshold 0.5**. Labels: `0=COVID-negative`, `1=COVID-positive`.

Frozen classifier: `C=0.1`, `solver=liblinear`, `class_weight=None`; the saved `.joblib` contains fitted coefficients and scaler. **Never retrain or retune during GUI inference.**

## 2. Packaged artifacts

| File | Role |
|---|---|
| `models/fusion/ct_b_xray_logistic_frozen.joblib` | **Final fitted scaler + classifier** |
| `models/ct/v61_13a378082055_best.pt` | CT mean-pooling encoder checkpoint |
| `models/xray/model_B_v2_27sep.keras` | X-ray B encoder |
| `models/xray/modelC_resnet50_18oct.keras` | X-ray C encoder |
| `models/side_detector/side_clf.joblib`, `invert_template.npy` | X-ray side/inversion QC |
| `config/xray_thresholds_train_only.json` | Fixed train-derived adaptive intensity thresholds |
| `config/ct_preprocessing_config.json` | CT V5 parameters (when available) |
| `src/` | **Reference code** for architecture, preprocessing and batch inference |
| `reports/freeze_report.json`, `reports/final_test_summary.json` | Model identity, hashes and evaluation record |
| `manifest.json` | SHA-256 and sizes of ZIP members |

If the packager lists optional missing files, request those before implementing the corresponding preprocessing step. The code files are **reference scripts**, not a production-ready API.

## 3. Preprocessing fidelity

### CT

Follow the teammate's V5 NIfTI pipeline: orientation handling, Lungmask R231 lung-guided crop/positioning, physical resampling to `(64,192,192)` with spacing `(5,2,2)` mm (Z,Y,X), HU clipping `[-1000,400]`, then scaling to `[0,1]`. Use the exact implementation and saved configuration, including QC. Reject unsuitable scans rather than forcing an output. The 2.5D encoder's exact view indices and pooling behavior must come from `src/ct_v61_frozen_baseline.py` / `src/train_ct_v61.py`, **not guessed**.

### X-ray

Only retained AP/PA images should be used. Reproduce the original side/inversion QC and its label-blind manual review procedure for uncertain cases. **Do not reuse historical patient-specific manual overrides for new uploads.**

- **Model C:** `robust` 1st–99th percentile intensity normalization, established inversion correction **before** resize-with-padding to 224×224; grayscale PNG → RGB float in `[0,1]`; extract the layer named `dense` (128D).
- **Model B:** use train-derived adaptive thresholds to select `direct`, `robust_1_99`, or `clahe_after_robust` (CLAHE clip limit 2.0, grid 8×8); apply established inversion correction before TensorFlow `resize_with_pad(224,224,antialias=False)`; replicate grayscale to 3 channels and divide by 255; extract layer `dense` (128D).
- Concatenate `[B128,C128]` for each retained X-ray; **mean aggregate image features**, not probabilities, by patient.

Train-derived thresholds (included in JSON): `mean_low=6.888329798410727`, `mean_high=82.86402995929011`, `std_low=2.3629952207195193`, `dark_high=96.90913179341574`.

## 4. GUI/API implementation recommendations

Implement a single-patient service with inputs: one CT `.nii`/`.nii.gz`, one or more X-ray images, and a temporary patient/session identifier. Suggested return fields: `status`, `positive_probability`, `predicted_class`, `threshold`, `ct_qc`, `xray_qc`, `accepted_xray_count`, and `model_version`. Report explicit QC failures; do not silently fall back to single-modality predictions, since the frozen classifier expects both modalities. Use model objects loaded once at service startup, not per request.

**Reference only:** `src/extract_ct_test_features.py`, `src/extract_xray_test_features_reviewed.py`, `src/evaluate_fusion_test.py` are batch/test utilities and may depend on manifests, paths, and held-out test labels. Adapt their inference functions; do not expose their evaluation logic to GUI users. Verify architecture loading on CPU and intended deployment device. Only deserialize trusted `.joblib`/`.pt` files.

### Integration acceptance checks

- Load all four model assets (CT, X-ray B, X-ray C, fusion) without changing weights.
- For a known, authorized **non-test** reference case, compare preprocessing and features against the original pipeline; validate `(1024,)`, `(128,)`, `(128,)`, `(256,)`, `(1280,)`, finite values and ordering.
- Confirm the GUI's `predict_proba` matches the frozen batch pipeline within numerical tolerance on the same approved inputs.
- Reject unreadable/side-view X-rays, invalid CT geometry, missing modality and unresolved QC; log reasons.
- Preserve input provenance and avoid storing raw patient images or identifiers in routine logs.

## 5. Reported results and limitations

Final held-out BIMCV **99-patient** test: **Macro F1 0.7475**, **ROC-AUC 0.8021**, **accuracy 74.75%**, COVID-positive recall **82.22%**, COVID-negative recall **68.52%**, confusion matrix `[[37,17],[8,37]]`. Validation (97 patients): Macro F1 **0.7726**, AUC **0.8040**. No threshold tuning after final test.

Modality matching was **patient-level**, not verified as same-session or time-synchronized. Results are **within-dataset**, not external or clinical validation. Validation fusion improvement over CT-only was modest and bootstrap uncertainty included zero. The test X-ray cleaning used label-blind review of 15 images and is not identical to the original training manual-review workflow. Confirm upstream pretrained encoder patient independence when making research claims.

## 6. Scope of ZIP and deployment dependencies

This ZIP intentionally excludes original BIMCV data, preprocessed patient tensors, cached patient features, patient manifests, manual-review CSVs, and patient-level predictions. It is **an integration handover**, not a self-contained, immediately executable GUI. The receiver needs a Python environment with compatible PyTorch/torchvision, TensorFlow/Keras, scikit-learn/joblib, nibabel, SimpleITK, lungmask, OpenCV, NumPy, pandas and Pillow; record exact versions after validating in the target environment. Some pretrained components may require separate runtime model downloads (e.g. side-detection backbone). Verify dataset/model licensing and institutional sharing policy before distributing checkpoints.

## 7. AI assistant handover prompt

> Integrate this frozen BIMCV CT + X-ray patient-level classifier into a research-only GUI. Read `HANDOVER.md` and the included Python source before coding. Preserve CT V5 preprocessing, CT V6.1 mean pooling, X-ray Model B/C encoder-specific preprocessing, B-before-C mean aggregation, CT-before-X-ray concatenation, and the saved scaler+Logistic Regression at threshold 0.5. Do not retrain or tune. Build an inference-only API with explicit QC handling. Validate feature shapes and numeric agreement against a non-test reference before GUI deployment. Flag missing dependencies and any ambiguity instead of approximating preprocessing.
