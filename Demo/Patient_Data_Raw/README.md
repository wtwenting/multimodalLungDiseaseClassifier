# Raw CT and X-ray GUI Demonstration Data

## 1. Overview

This directory contains raw CT and chest X-ray inputs for five previously evaluated BIMCV patients, intended for GUI integration and regression testing.

These samples are not an independent test dataset and must not be used for model selection, threshold tuning or clinical validation.

## 2. Directory Structure

```text
Demo/Patient_Data_Raw/
├── README.md
├── demo_manifest.csv
├── checksums.json
├── ct_reference_features_1024d.npy
├── xray_reference_features_256d.npy
├── patient_01/
│   ├── scan.nii.gz
│   ├── image_01.png
│   ├── image_02.png
│   └── image_03.png
├── patient_02/
│   ├── scan.nii.gz
│   └── image_01.png ... image_07.png
├── patient_03/
│   ├── scan.nii.gz
│   └── image_01.png
├── patient_04/
│   ├── image_01.png
│   └── scan.nii.gz  (only if added and verified)
└── patient_05/
    ├── scan.nii.gz
    └── image_01.png
```

All raw CT and X-ray files are stored directly inside each patient directory. There are no separate `ct/` or `xray/` subdirectories.

## 3. Patient Data Availability

| Patient | Raw CT | Raw X-rays |
|---|---|---|
| patient_01 | Available | 3 |
| patient_02 | Available | 7 |
| patient_03 | Available | 1 |
| patient_04 | Verify locally | 1 |
| patient_05 | Available | 1 |

The inspected repository revision did not contain `patient_04/scan.nii.gz`. If the file has subsequently been added, confirm that it belongs to the patient identified in `demo_manifest.csv` and matches the intended CT preprocessing source.

## 4. Raw Image Processing

**CT processing**

1. Load the NIfTI CT volume (`scan.nii.gz`).
2. Apply the frozen V5 CT preprocessing configuration.
3. Prepare the 12-view CT Encoder B input.
4. Extract and aggregate the CT representation into a 1,024-dimensional patient-level feature vector.

**X-ray processing**

1. Load all `image_NN.png` files belonging to the patient.
2. Apply the original X-ray Model B and Model C preprocessing workflows.
3. Extract 128-dimensional features from each encoder.
4. Concatenate Model B followed by Model C features.
5. Mean-aggregate across the retained X-ray images to obtain a 256-dimensional patient-level vector.

Any label-blind manual quality-control decisions required by the original workflow must be reproduced for strict feature comparison.

## 5. Fusion and Reference Results

Concatenate the patient-level feature vectors in this order:

CT (1,024D) + X-ray (256D) = 1,280D.

Apply the previously fitted `StandardScaler` and frozen Logistic Regression classifier with decision threshold 0.5. Do not refit the scaler or retrain the classifier using the demo patients.

The reference feature arrays contain five rows in the order specified by `demo_manifest.csv`. Reference probabilities and predictions are provided for regression comparison, not model tuning.

## 6. Validation and Limitations

- Verify patient-to-file associations using the manifest.
- Verify CT and X-ray file readability before inference.
- Confirm raw CT availability for `patient_04` before attempting raw CT inference.
- Use all retained X-rays when reproducing the original patient-level aggregation.
- CT and X-ray inputs are patient-linked, but are not confirmed to be from the same imaging session.
- Respect BIMCV redistribution permissions and medical-data privacy requirements.

**This package is for research demonstration only and is not a clinically validated diagnostic system.**