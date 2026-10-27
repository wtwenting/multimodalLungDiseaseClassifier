# Model A data preparation (BIMCV-COVID19, X-ray-only patients, frontal views).
# Runs locally (VS Code / Jupyter) where the downloaded images are. Each "# %%" block is one
# notebook cell, run top to bottom. Output: C:\BIMCV_xray\modelA_224 (zip it for Colab).
#
# Order: Step 1 manifest -> Step 3 convert + split -> Step 6a audit (optional)
#        -> Step 6c review new images -> Step 6d apply decisions -> zip
#
# The first-batch cleaning (steps 3b-3h: side-view detector calibration, inverted rule,
# manual drops) lives in notebook version 1. Its decisions are saved by filename in
# modelA_224\review_auto\cleaning_decisions.csv, which 6c and 6d read.

# %% STEP 1: build one manifest of all downloaded BIMCV X-rays (one row per image)
import re
from pathlib import Path
import pandas as pd
from PIL import Image

ROOTS = [                                   # every download folder
    Path(r"C:\BIMCV_xray"),
    Path(r"C:\BIMCV_xray_pos2"),
    Path(r"C:\BIMCV_xray_neg"),
    Path(r"C:\BIMCV_xray_neg2"),
]
MANIFEST_PATH = Path(r"C:\BIMCV_xray\modelA_manifest.csv")

rows = []
for root in ROOTS:
    for p in (root / "data").glob("covid*/xray/sub-*/*/*.png"):
        name = p.name.lower()
        view = re.search(r"_vp-([a-z]+)_", name)
        with Image.open(p) as im:                       # reads the header only
            mode, (w, h) = im.mode, im.size
        rows.append({
            "filename": p.name,
            "filepath": str(p),
            "patient_id": re.search(r"sub-(s\d+)", name).group(1).upper(),
            "session": p.parent.name,
            "view": view.group(1).upper() if view else "UNKNOWN",
            "class": "COVID" if p.parts[-5] == "covidPositive" else "Negative",
            "source_root": str(root),
            "mode": mode, "width": w, "height": h,
        })

df = pd.DataFrame(rows)
before = len(df)
df = df.drop_duplicates(subset=["patient_id", "filename"]).reset_index(drop=True)   # same image in two folders
df["label"] = (df["class"] == "COVID").astype(int)                                 # COVID = 1, Negative = 0

print(f"Images: {len(df)}  (removed {before - len(df)} duplicates across folders)")
print("\nPatients per class:\n", df.groupby("class")["patient_id"].nunique())
print("\nImages per class / view:\n", df.groupby(["class", "view"]).size())
print("\nPatients in more than one class:", (df.groupby("patient_id")["class"].nunique() > 1).sum())
print("\nImage modes (I;16 / I = 16-bit, L = 8-bit):\n", df["mode"].value_counts())
print("\nMedian size:", int(df["width"].median()), "x", int(df["height"].median()))

df.to_csv(MANIFEST_PATH, index=False)
print("\nSaved:", MANIFEST_PATH)

# %% STEP 3: robust 1-99 -> resize_with_pad 224 -> 8-bit PNG, plus patient-level split
import hashlib
import numpy as np
import tensorflow as tf

OUT_DIR = Path(r"C:\BIMCV_xray\modelA_224")       # small training-ready copy (zip this later)
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# --- same functions as groupmate's benchmark_adaptive_xray.py ---
def read_image(path):
    with Image.open(path) as im:
        arr = np.asarray(im)
        if arr.ndim == 3:
            arr = np.asarray(im.convert("L"))
    return arr.astype(np.float32)

def direct(a):
    if a.max() > 255:
        a = np.floor(a / 256.0)
    return np.clip(a, 0, 255).astype(np.uint8)

def robust(a):
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return direct(a)
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

def to_224(a8):                      # same resize as groupmate's make_tensor, kept 1 channel / 0-255
    x = tf.convert_to_tensor(a8[..., None], dtype=tf.float32)
    x = tf.image.resize_with_pad(x, 224, 224, antialias=False)
    return np.clip(np.rint(x.numpy()[..., 0]), 0, 255).astype(np.uint8)

# --- patient-level split from a hash of the patient ID (stable when patients are added) ---
def split_for(pid):
    h = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"

df = pd.read_csv(MANIFEST_PATH)
df["split"] = df["patient_id"].map(split_for)
df["out_name"] = df["filename"]                                   # BIMCV names already include sub-/ses-
df["rel_path"] = "images/" + df["out_name"]

done = 0
for i, r in enumerate(df.itertuples(), 1):
    out = IMG_DIR / r.out_name
    if not out.exists():                                          # rerun = only new images
        Image.fromarray(to_224(robust(read_image(r.filepath)))).save(out)
        done += 1
    if i % 200 == 0 or i == len(df):
        print(f"{i}/{len(df)} checked, {done} newly converted")

cols = ["rel_path", "filename", "patient_id", "session", "view", "class", "label", "split"]
df[cols].to_csv(OUT_DIR / "modelA_224_manifest.csv", index=False)

print("\nPatients per split / class:\n", df.groupby(["split", "class"])["patient_id"].nunique().unstack())
print("\nImages per split / class:\n", df.groupby(["split", "class"]).size().unstack())
print("\nPatient in more than one split:", (df.groupby("patient_id")["split"].nunique() > 1).sum())
print("Saved:", OUT_DIR)

# %% STEP 6a (optional): audit all download folders (read-only)
rows = []
for root in ROOTS:
    for p in root.glob("data/covid*/xray/sub-*/*/*.png"):
        rows.append({"root": root.name, "path": str(p), "filename": p.name,
                     "patient_id": p.parts[-3],
                     "class": "COVID" if "posi" in p.parts[-5].lower() else "Negative"})
files = pd.DataFrame(rows)

print("Images per folder / class:\n", files.groupby(["root", "class"]).size().unstack(fill_value=0))
uniq = files.drop_duplicates(["patient_id", "filename"])
print("\nAfter removing cross-folder duplicates:", len(uniq), "images")
print(uniq.groupby("class").agg(patients=("patient_id", "nunique"), images=("filename", "size")))

bad = []
for p in uniq["path"]:
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception:
        bad.append(p)
print("\nUnreadable files:", len(bad), bad[:5])

# %% STEP 6c: check only the new images; old decisions carry over by filename
from PIL import ImageDraw
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REVIEW = OUT_DIR / "review_auto"
m   = pd.read_csv(OUT_DIR / "modelA_224_manifest.csv")
dec = pd.read_csv(REVIEW / "cleaning_decisions.csv")
m = m.merge(dec[["filename", "keep", "drop_side", "invert"]], on="filename", how="left")
m["is_new"] = ~m["filename"].isin(dec["filename"])
print("old:", (~m["is_new"]).sum(), "| new:", m["is_new"].sum(), m[m["is_new"]].groupby("class").size().to_dict())

# 1) inverted rule (as in 3d), with the template built from the kept old images
S = 64
def crop64(p):
    a = np.asarray(Image.open(p), dtype=np.float32)
    ys, xs = np.where(a > 10)
    if len(ys) == 0:
        return 0.0, np.zeros((S, S), np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = np.asarray(Image.fromarray(a[y0:y1, x0:x1]).resize((S, S)), dtype=np.float32)
    return (y1 - y0) * (x1 - x0) / a.size, (c - c.mean()) / (c.std() + 1e-6)

fracs, crops = zip(*(crop64(OUT_DIR / p) for p in m["rel_path"]))
crops = np.stack(crops)
m["content_frac"] = fracs
old_keep = (~m["is_new"] & m["keep"].eq(True)).values
t = np.median(crops[old_keep], axis=0)
t = (t - t.mean()) / t.std()
m["template_corr"] = (crops * t).mean(axis=(1, 2))
m["auto_invert"] = m["is_new"] & (m["template_corr"] < 0)

# 2) side-view detector, retrained on the earlier decisions
base = tf.keras.applications.MobileNetV2(include_top=False, pooling="avg",
                                         input_shape=(224, 224, 3), weights="imagenet")
def load(path, inv):
    x = tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)
    x = tf.where(inv, 255.0 - x, x)
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)

ds = tf.data.Dataset.from_tensor_slices(([str(OUT_DIR / p) for p in m["rel_path"]],
                                         m["auto_invert"].values)).map(load).batch(64)
emb = base.predict(ds, verbose=1)

is_side, is_front = m["drop_side"].eq(True), m["keep"].eq(True)
use = (~m["is_new"] & (is_side | is_front)).values
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
clf.fit(emb[use], is_side[use].astype(int))
m["p_side"] = clf.predict_proba(emb)[:, 1]

# 3) the new images, side views first, with flags
new = m[m["is_new"]].sort_values("p_side", ascending=False).reset_index(drop=True)
new["new_id"] = new.index
flag_parts = [np.where(new["p_side"] > 0.5, "SIDE ", ""),
              np.where(new["auto_invert"], "INV ", ""),
              np.where(new["content_frac"] < 0.6, "PAD", "")]
new["flags"] = ["".join(f).strip() for f in zip(*flag_parts)]
new.to_csv(REVIEW / "new_review.csv", index=False)
print("SIDE:", (new["p_side"] > 0.5).sum(), "| uncertain (0.2-0.8):", new["p_side"].between(0.2, 0.8).sum(),
      "| INV:", new["auto_invert"].sum(), "| PAD:", (new["content_frac"] < 0.6).sum())

def sheets(rows, N=10, TILE=128):
    for s in range(0, len(rows), N * N):
        chunk = rows.iloc[s:s + N * N]
        img = Image.new("L", (N * TILE, -(-len(chunk) // N) * TILE))
        draw = ImageDraw.Draw(img)
        for k, r in enumerate(chunk.itertuples()):
            x, y = (k % N) * TILE, (k // N) * TILE
            img.paste(Image.open(OUT_DIR / r.rel_path).resize((TILE, TILE)), (x, y))
            label = f"{r.new_id} {r.flags}"
            draw.rectangle([x, y, x + 7 * len(label), y + 12], fill=0)
            draw.text((x + 2, y), label, fill=255)
        display(img)

sheets(new)

# %% STEP 6c (extra): show the uncertain side-view scores and the padded images
unsure = new[new["p_side"].between(0.2, 0.8)]
sheets(unsure.assign(flags=unsure["flags"] + " p=" + unsure["p_side"].round(2).astype(str)))

pad = new[new["content_frac"] < 0.6]
sheets(pad)

# %% STEP 6d: apply decisions to the new images, then write ONE clean manifest for all images
# new_id lists from the 6c sheets (final run: 3 uncertain + 4 padded + 3 picked by eye)
DROP     = list(unsure["new_id"]) + list(pad["new_id"]) + [77, 106, 291]
NOT_SIDE = []
NOT_INV  = []

new["drop_side"]   = (new["p_side"] > 0.5) & ~new["new_id"].isin(NOT_SIDE)
new["drop_manual"] = new["new_id"].isin(DROP)
new["invert"]      = new["auto_invert"] & ~new["new_id"].isin(NOT_INV)
new["keep"]        = ~(new["drop_side"] | new["drop_manual"])

# flip the inverted new images from the original file (255 - robust before resize keeps padding black)
orig = pd.read_csv(MANIFEST_PATH)[["filename", "filepath"]]
fix = new[new["keep"] & new["invert"]].merge(orig, on="filename")
for r in fix.itertuples():
    Image.fromarray(to_224(255 - robust(read_image(r.filepath)))).save(OUT_DIR / r.rel_path)
print("Flipped new images:", len(fix))

# add the new decisions to the decision record (backup first), so future reruns treat them as old
dec.to_csv(REVIEW / "cleaning_decisions_backup_before_6d.csv", index=False)
keep_cols = ["filename", "keep", "drop_side", "drop_manual", "invert", "p_side", "auto_invert", "content_frac"]
all_dec = pd.concat([dec, new[keep_cols]], ignore_index=True)
all_dec.to_csv(REVIEW / "cleaning_decisions.csv", index=False)

# one clean manifest for everything
m2 = pd.read_csv(OUT_DIR / "modelA_224_manifest.csv").merge(all_dec[["filename", "keep", "invert"]], on="filename", how="left")
print("Images without a decision (should be 0):", m2["keep"].isna().sum())
clean = m2[m2["keep"].eq(True)]
cols = ["rel_path", "filename", "patient_id", "session", "view", "class", "label", "split", "invert"]
clean[cols].to_csv(OUT_DIR / "modelA_224_clean_manifest.csv", index=False)

print("\nClean images:", len(clean))
print("Images per split / class:\n", clean.groupby(["split", "class"]).size().unstack())
print("\nPatients per split / class:\n", clean.groupby(["split", "class"])["patient_id"].nunique().unstack())

sheets(fix.assign(flags="fixed"))                # check the newly flipped ones look normal

# %% ZIP for Colab (upload to MyDrive/PSUPRProject_2026Sem2/bimcv_xray/)
import shutil
shutil.make_archive(r"C:\BIMCV_xray\modelA_V2_224", "zip", str(OUT_DIR))
