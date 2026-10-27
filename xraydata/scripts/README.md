# Model A data: BIMCV-COVID19 X-ray download, preprocessing and split

COVID vs Negative chest X-rays from BIMCV-COVID19, X-ray-only patients (no CT), frontal (AP/PA) views only.

## 1. Download (`download/`)
1. `make_frontal_plan.py --work C:\BIMCV_xray\xray_work --positive 1000 --negative 1000`
   picks X-ray-only patients with a frontal image and writes `frontal_xray_only_plan.csv`.
2. `download_extract_xray.py` downloads one archive at a time, extracts only the planned
   patients' X-rays, verifies, deletes the archive and repeats. Safe to stop and rerun.

Downloaded into four roots, merged in step 1 below:
`C:\BIMCV_xray`, `C:\BIMCV_xray_pos2`, `C:\BIMCV_xray_neg`, `C:\BIMCV_xray_neg2`.
Final: COVID 1,013 patients / 2,194 images, Negative 1,036 / 1,639 (3,833 images).

## 2. Preprocess, split and clean (`prep/prep_modelA.py`)
Notebook-style cells (`# %%`), run top to bottom in VS Code or Jupyter on the PC holding the images.

| Step | What it does |
|---|---|
| 1 | Manifest of every image across the four roots, duplicates dropped (`modelA_manifest.csv`) |
| 3 | Robust 1-99 percentile stretch per image on the full-size original, `resize_with_pad` 224, 8-bit PNG. Patient-level split from `md5(patient_id) % 100`: <70 train, <85 val, else test |
| 6a | Optional audit of the download folders |
| 6c | Flags new images: side view (MobileNetV2 features + logistic regression trained on earlier decisions), inverted (`template_corr < 0`), padded (`content_frac < 0.6`) |
| 6d | Applies drops and flips, appends to `cleaning_decisions.csv`, writes `modelA_224_clean_manifest.csv` |
| zip | `modelA_V2_224.zip` for Colab |

Why these choices:
- **Robust 1-99, not /256:** most files are 12-bit stored in 16-bit, and the share differs by class, so `/256` made Negatives darker (a class shortcut).
- **Hash split:** adding patients never moves existing ones between splits, and no patient is in two splits.
- **Frontal only:** about 19% of images tagged PA/AP were actually side views.
- **Inverted images flipped:** almost all were Negative, another class shortcut. They are re-converted as `255 - robust(original)` before resizing, so the padding stays black.

The first batch was cleaned in notebook version 1 (steps 3b to 3h). Those decisions, and every later one, are stored by filename in `modelA_224/review_auto/cleaning_decisions.csv`, which is the record that steps 6c and 6d build on.

Result: all 3,833 images have a decision; the kept ones are listed in `modelA_224_clean_manifest.csv`.
