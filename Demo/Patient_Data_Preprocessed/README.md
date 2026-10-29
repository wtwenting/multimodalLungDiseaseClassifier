# Preprocessed CT + X-ray GUI demo (5 previously evaluated test patients)

No preprocessing is needed in the GUI. These are **encoder-ready arrays**, not raw medical images.

- `patient_XX/ct/ct_v5_preprocessed.npz`: key `volume`, shape (64,192,192), float32, V5-preprocessed. Use the original 12-view CT Encoder B preparation (`make_views`) before DenseNet121.
- `patient_XX/xray/image_NN_model_b_input.npy`: shape (224,224,3), float32 [0,1], encoder B input.
- `patient_XX/xray/image_NN_model_c_input.npy`: shape (224,224,3), float32 [0,1], encoder C input.
- For each image extract 128D from B `dense` and 128D from C `dense`, concatenate **B then C**, mean across all retained images to obtain 256D per patient.
- CT Encoder B mean-pooled features are 1024D. Concatenate **CT first then X-ray** (1280D) and use the **frozen fitted StandardScaler + LogisticRegression** with threshold 0.5. Do not refit or retrain.
- `ct_reference_features_1024d.npy` and `xray_reference_features_256d.npy` are (5,1024) and (5,256), respectively, ordered by `demo_manifest.csv`. Check GUI feature vectors and probabilities against these references; small float32 differences may occur.
- Reference predictions are from a previously evaluated held-out test cohort; this is a **GUI regression demo, not new independent evaluation**. Do not tune models using these cases.
- Source: BIMCV; check data redistribution rules. CT and X-ray are patient-matched but not confirmed same-session. Research use only, not clinical diagnosis.
