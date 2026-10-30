# CT + X-ray GUI demonstration (5 previously evaluated test patients)

This package is for GUI regression/integration demonstration, **not new independent evaluation**.

Each `patient_XX/ct/scan.nii.gz` (or `.nii`) is the original CT selected by the V5 preprocessing manifest. Each `patient_XX/xray/` contains the original X-rays ACCEPTed in the reviewed test QC. All included X-rays must be used for mean aggregation to reproduce the reference feature vector.

`demo_manifest.csv` maps demo IDs to source patient IDs and includes previously recorded labels, probabilities and predicted classes. **Do not use these reference outputs to tune the model or QC policy.**

Reference arrays have five rows in exactly the order of `demo_manifest.csv`: CT (5,1024), X-ray (5,256). To validate the fusion stage, concatenate CT first and X-ray second, then run the existing frozen StandardScaler + LogisticRegression pipeline with threshold 0.5.

For full end-to-end testing, preprocess the raw images with the frozen V5 CT and original X-ray B/C workflows and compare extracted features to the references. Some original X-rays required label-blind manual QC; the GUI may require the same review decisions to reproduce outputs. CT/X-ray images are patient-linked, not confirmed same-session.

Data source: BIMCV. Verify dataset license and redistribution permissions before sharing raw images. Research demonstration only; not for clinical diagnosis.
