# Preprocessed CT and X-ray GUI Demonstration Data

## 1. Overview

This directory contains encoder-ready CT and X-ray arrays for five previously evaluated BIMCV patients.

The files have already undergone the selected preprocessing workflows and are intended for GUI integration testing without repeating raw medical-image preprocessing.

This package is for regression testing, not independent model evaluation.

## 2. Directory Structure

```text
Demo/Patient_Data_Preprocessed/
├── README.md
├── demo_manifest.csv
├── checksums.json
├── ct_reference_features_1024d.npy
├── xray_reference_features_256d.npy
├── verify_preprocessed.py
├── patient_01/
│   ├── ct_v5_preprocessed.npz
│   ├── image_01_model_b_input.npy
│   ├── image_01_model_c_input.npy
│   └── ...
├── patient_02/
├── patient_03/
├── patient_04/
│   ├── ct_v5_preprocessed.npz
│   ├── image_01_model_b_input.npy
│   └── image_01_model_c_input.npy
└── patient_05/
```

The `.npz` and `.npy` files are stored directly inside each patient directory, without separate `ct/` or `xray/` subdirectories.

## 3. Preprocessed Input Specifications

| Modality | File | Expected format |
|---|---|---|
| CT | `ct_v5_preprocessed.npz` | Key `volume`, shape `(64, 192, 192)`, float32 |
| X-ray Model B | `image_NN_model_b_input.npy` | Shape `(224, 224, 3)`, float32, range [0, 1] |
| X-ray Model C | `image_NN_model_c_input.npy` | Shape `(224, 224, 3)`, float32, range [0, 1] |

The CT volume still requires the original 12-view preparation (`make_views`) before entering CT Encoder B.

## 4. Feature Extraction

**CT Encoder B**

Process the preprocessed CT volume through the original CT Encoder B preparation and frozen encoder, producing a 1,024-dimensional patient-level feature vector.

**X-ray Encoders B and C**

For each X-ray image, extract 128-dimensional features from Model B and 128-dimensional features from Model C.

Concatenate Model B followed by Model C, then mean-aggregate across all retained images to produce a 256-dimensional patient-level X-ray feature vector.

## 5. Frozen Fusion Classifier

The selected fusion input is:

CT (1,024D) + X-ray (256D) = 1,280D.

Apply the existing fitted `StandardScaler` and Logistic Regression classifier with threshold 0.5.

Do not refit the scaler, modify the feature order or retrain the classifier.

## 6. Reference Verification

The files `ct_reference_features_1024d.npy` and `xray_reference_features_256d.npy` contain reference feature arrays of shapes `(5, 1024)` and `(5, 256)` respectively.

Their row order follows `demo_manifest.csv`.

To verify the packaged inputs, run from the repository root:

```bash
python Demo/Patient_Data_Preprocessed/verify_preprocessed.py
```

Compare newly extracted features and predictions against the recorded references. Small floating-point differences may occur across compatible environments.

All five patients have preprocessed CT and X-ray inputs in the inspected repository revision, including `patient_04`.

## 7. Limitations

- The five patients were previously evaluated; they are not a new independent test cohort.
- The reference labels and predictions must not be used for model tuning.
- CT and X-ray data are patient-matched, not confirmed same-session.
- Original model checkpoints and compatible preprocessing implementations are required for strict reproduction.
- BIMCV data-use and redistribution restrictions must be respected.

**This package is intended for research demonstration and software integration testing, not clinical diagnosis.**