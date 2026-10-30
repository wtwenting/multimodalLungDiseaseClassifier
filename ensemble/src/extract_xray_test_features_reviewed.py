#!/usr/bin/env python3
"""Frozen Model B v2 + Model C test-only X-ray 256D extraction.
Uses existing train-derived adaptive thresholds and original cleaning functions.
No labels are read for extraction. Uncertain side/inversion cases are excluded
pending label-blind review; no manual overrides are guessed.
Run from ensemble/, with clean_modelc_xray.py and compare_encoders.py in src/.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from clean_modelc_xray import robust, read_image, resize224, crop64
from compare_encoders import model_b_pixels
from audit_adaptive_xray import stats, classify


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('output/xray_overnight/splits/test_manifest.csv'))
    p.add_argument('--thresholds', type=Path, default=Path('output/xray_adaptive_audit/thresholds_train_only.json'))
    p.add_argument('--detector', type=Path, default=Path('models/side_detector'))
    p.add_argument('--model-b', type=Path, default=Path('../xraydata/Models/ModelB_CovidvsNormal_27sep/model_B_v2_27sep.keras'))
    p.add_argument('--model-c', type=Path, default=Path('../xraydata/Models/modelC_resnet50_18oct.keras'))
    p.add_argument('--out', type=Path, default=Path('output/xray_test_features'))
    p.add_argument('--smoke', action='store_true', help='Inspect at most 3 test images; separate smoke output')
    p.add_argument('--manual-review', type=Path, default=None, help='Approved label-blind review CSV for REVIEW images')
    p.add_argument('--review-policy', choices=['exclude-uncertain'], default='exclude-uncertain')
    args = p.parse_args()

    # Only inspect paths and non-label metadata from test manifest.
    df = pd.read_csv(args.manifest, dtype={'key': str, 'path': str})
    required = {'key', 'path', 'view'}
    if not required.issubset(df):
        raise ValueError(f'Missing manifest fields: {required-set(df)}')
    if 'split' in df and not df['split'].astype(str).str.lower().eq('test').all():
        raise ValueError('Input must be test-only')
    df = df[df['view'].astype(str).str.upper().isin(('AP', 'PA'))].copy()
    df['patient_id'] = df['key'].astype(str).str.split(':').str[-1].str.upper()
    if df.empty:
        raise ValueError('No AP/PA test images')
    if df['path'].duplicated().any():
        raise ValueError('Duplicate test image paths')
    if df.groupby('patient_id')['key'].nunique().max() != 1:
        raise ValueError('Ambiguous patient key mapping')
    df = df.sort_values(['patient_id', 'path']).reset_index(drop=True)
    if args.smoke:
        df = df.head(3)
    for f in (args.thresholds, args.detector/'side_clf.joblib', args.detector/'invert_template.npy', args.model_b, args.model_c):
        if not f.is_file():
            raise FileNotFoundError(f)
    thresholds = json.loads(args.thresholds.read_text())
    if set(thresholds) != {'mean_low', 'mean_high', 'std_low', 'dark_high'}:
        raise ValueError('Unexpected adaptive threshold keys')
    template = np.load(args.detector/'invert_template.npy', allow_pickle=False)
    if template.shape != (64,64):
        raise ValueError('Wrong inversion template shape')
    clf = joblib.load(args.detector/'side_clf.joblib')  # Trusted project file only.
    if 1 not in list(clf.classes_):
        raise ValueError('Side classifier missing class 1')
    import cv2
    import tensorflow as tf
    from tensorflow import keras
    base = keras.applications.MobileNetV2(include_top=False, pooling='avg', input_shape=(224,224,3), weights='imagenet')
    full_b = keras.models.load_model(args.model_b, compile=False)
    full_c = keras.models.load_model(args.model_c, compile=False)
    if tuple(full_b.input_shape[1:]) != (224,224,3):
        raise ValueError('Model B unexpected input')
    enc_b = keras.Model(full_b.inputs, full_b.get_layer('dense').output)
    enc_c = keras.Model(full_c.inputs, full_c.get_layer('dense').output)
    enc_b.trainable = False
    enc_c.trainable = False

    # Load review decisions without inspecting test labels. Every review row must
    # match the recomputed patient ID and exact original image path.
    reviews = {}
    if args.manual_review is not None:
        manual = pd.read_csv(args.manual_review, dtype={'patient_id':str, 'source_path':str})
        needed = {'patient_id', 'source_path', 'manual_decision'}
        if not needed.issubset(manual.columns):
            raise ValueError(f'Manual review missing columns: {needed-set(manual.columns)}')
        if manual.source_path.duplicated().any():
            raise ValueError('Duplicate manual review source paths')
        allowed = {'ACCEPT', 'ACCEPT_INVERT', 'EXCLUDE_SIDE', 'EXCLUDE_QUALITY', 'UNRESOLVED'}
        if not manual.manual_decision.isin(allowed).all():
            raise ValueError('Unrecognized manual decision')
        for r in manual.itertuples(index=False):
            reviews[str(r.source_path)] = (str(r.patient_id).upper(), str(r.manual_decision))
        if args.smoke:
            raise ValueError('Do not combine --manual-review with --smoke')
        input_paths = set(df.path.astype(str))
        if not set(reviews).issubset(input_paths):
            raise ValueError('Manual review contains paths not in test manifest')

    out = args.out/'smoke' if args.smoke else args.out
    out.mkdir(parents=True, exist_ok=True)
    image_rows = []
    vectors = []
    applied_reviews = set()
    for i, row in enumerate(df.itertuples(index=False), 1):
        source = Path(row.path)
        if not source.is_file():
            raise FileNotFoundError(source)
        # Exact detector staging: robust grayscale -> resize_with_pad 224 ->
        # content crop/template correlation -> MobileNetV2 embedding on inverted candidate.
        raw = read_image(source)
        candidate = resize224(robust(raw))
        # crop64 expects a path; use a temporary 224 PNG to preserve the original helper.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp)/'candidate.png'
            Image.fromarray(candidate).save(png)
            frac, crop = crop64(png)
        corr = float((crop * template).mean())
        invert = corr < 0
        mobilenet_input = np.repeat(candidate[...,None],3,axis=-1).astype(np.float32)
        if invert:
            mobilenet_input = 255.0 - mobilenet_input
        mobilenet_input = keras.applications.mobilenet_v2.preprocess_input(mobilenet_input)
        embedding = np.asarray(base(mobilenet_input[None,...], training=False))
        side_probability = float(clf.predict_proba(embedding)[0, list(clf.classes_).index(1)])
        review = (0.2 <= side_probability <= 0.8) or invert or (frac < 0.6)
        drop_side = side_probability > 0.5
        quality = classify(stats(raw), thresholds)
        decision = {'normal':'direct', 'dark':'robust_1_99', 'bright':'robust_1_99', 'low_contrast':'clahe_after_robust'}[quality]
        record = {'patient_id': row.patient_id, 'key': row.key, 'source_path': str(source),
                  'quality_flag': quality, 'selective_clahe_decision': decision,
                  'inversion_detected': bool(invert), 'side_probability': side_probability,
                  'content_frac': float(frac), 'review_required': bool(review),
                  'status': 'EXCLUDE_SIDE' if drop_side else 'REVIEW' if review else 'ACCEPT'}
        if str(source) in reviews:
            expected_patient, manual_decision = reviews[str(source)]
            if expected_patient != row.patient_id:
                raise ValueError(f'Manual patient mismatch for {source}')
            if record['status'] != 'REVIEW':
                raise ValueError(f'Manual review refers to non-REVIEW image: {source}')
            if manual_decision == 'ACCEPT_INVERT' and not invert:
                raise ValueError(f'ACCEPT_INVERT inconsistent with detector: {source}')
            if manual_decision == 'ACCEPT' and invert:
                raise ValueError(f'ACCEPT inconsistent with detected inversion: {source}')
            record['manual_decision'] = manual_decision
            record['status'] = ('ACCEPT_MANUAL' if manual_decision in ('ACCEPT','ACCEPT_INVERT')
                                else 'EXCLUDE_SIDE' if manual_decision == 'EXCLUDE_SIDE'
                                else 'EXCLUDE_QUALITY' if manual_decision == 'EXCLUDE_QUALITY'
                                else 'REVIEW')
            applied_reviews.add(str(source))
        else:
            record['manual_decision'] = ''
        if record['status'] in ('ACCEPT', 'ACCEPT_MANUAL'):
            # Finalized clean Model C PNG: inversion before resize, normalize 0-1.
            cleaned = robust(raw)
            if invert:
                cleaned = 255-cleaned
            c_png = resize224(cleaned)
            c_tensor = np.repeat(c_png[...,None],3,axis=-1).astype(np.float32)/255.0
            c_vec = np.asarray(enc_c(c_tensor[None,...], training=False)).reshape(-1)
            # Original B path: selective CLAHE on original pixels, inversion before padding.
            b_row = SimpleNamespace(orig_path=str(source), invert_final=invert)
            b_pixels = model_b_pixels(b_row, {'selective_clahe_decision':decision}, cv2)
            b_tensor = tf.image.resize_with_pad(tf.convert_to_tensor(b_pixels[...,None],tf.float32),224,224,antialias=False)
            b_tensor = tf.repeat(b_tensor,3,axis=-1)/255.0
            b_vec = np.asarray(enc_b(b_tensor[None,...], training=False)).reshape(-1)
            if c_vec.shape != (128,) or b_vec.shape != (128,):
                raise ValueError(f'Unexpected encoder dimensions: {c_vec.shape}, {b_vec.shape}')
            vec = np.concatenate([b_vec, c_vec]).astype(np.float32)
            if not np.isfinite(vec).all():
                raise ValueError('Nonfinite feature vector')
            vectors.append((row.patient_id, vec))
        image_rows.append(record)
        if i % 25 == 0 or i == len(df):
            print(f'X-ray test images {i}/{len(df)}', flush=True)

    if reviews and applied_reviews != set(reviews):
        raise ValueError('Not all manual reviews were applied')
    pd.DataFrame(image_rows).to_csv(out/'test_image_qc.csv', index=False)
    # Strict patient policy: exclude patients with any REVIEW image, to avoid
    # silently replacing manually reviewed training policy with automatic acceptance.
    qc = pd.DataFrame(image_rows)
    uncertain = set(qc.loc[qc.status.eq('REVIEW'), 'patient_id'])
    by_patient = {}
    for pid, vec in vectors:
        if pid not in uncertain:
            by_patient.setdefault(pid, []).append(vec)
    if not by_patient:
        raise RuntimeError('No accepted test patients; inspect test_image_qc.csv')
    patients = sorted(by_patient)
    feats = np.stack([np.mean(by_patient[pid],axis=0).astype(np.float32) for pid in patients])
    np.save(out/'xray_test_features_256d.npy', feats, allow_pickle=False)
    pd.DataFrame([{'patient_id': pid, 'split':'test', 'image_count':len(by_patient[pid])} for pid in patients]).to_csv(out/'xray_test_index.csv', index=False)
    report = {'status':'FEATURES_ONLY', 'test_images_examined':len(df), 'patients_with_features':len(patients),
              'image_qc_counts':qc.status.value_counts().to_dict(), 'patients_quarantined_for_review':len(uncertain),
              'model_b_sha256':file_hash(args.model_b), 'model_c_sha256':file_hash(args.model_c),
              'adaptive_thresholds_sha256':file_hash(args.thresholds),
              'model_b_first':True, 'model_c_second':True, 'feature_dim':256,
              'manual_review_file_sha256':file_hash(args.manual_review) if args.manual_review else None,
              'manual_review_applied':len(applied_reviews),
              'test_labels_read':False, 'test_predictions_generated':False, 'test_metrics_computed':False,
              'warning':'Test QC is label-blind and manually approved for reviewed images, but differs from original training override process; record this limitation.'}
    (out/'readiness.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
