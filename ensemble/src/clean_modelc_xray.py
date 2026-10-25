#!/usr/bin/env python3
"""Reproduce Model C X-ray cleaning on existing BIMCV split manifests.

Stages: prepare (convert + score + contact sheets), finalize (manual overrides), features.
Original PNGs are never modified. Test images are inventory-only by default.
"""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def direct(a):
    if a.max() > 255:
        a = np.floor(a / 256.0)
    return np.clip(a, 0, 255).astype(np.uint8)


def robust(a):
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return direct(a)
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def read_image(path):
    with Image.open(path) as im:
        arr = np.asarray(im)
        if arr.ndim == 3:
            arr = np.asarray(im.convert('L'))
    return arr.astype(np.float32)


def resize224(a):
    import tensorflow as tf
    x = tf.convert_to_tensor(a[..., None], dtype=tf.float32)
    x = tf.image.resize_with_pad(x, 224, 224, antialias=False)
    return np.clip(np.rint(x.numpy()[..., 0]), 0, 255).astype(np.uint8)


def crop64(path):
    a = np.asarray(Image.open(path), dtype=np.float32)
    ys, xs = np.where(a > 10)
    if not len(ys):
        return 0., np.zeros((64, 64), np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.asarray(Image.fromarray(a[y0:y1, x0:x1]).resize((64, 64)), dtype=np.float32)
    return float((y1-y0)*(x1-x0)/a.size), (crop-crop.mean())/(crop.std()+1e-6)


def load_manifests(folder):
    frames = []
    for split in ('train', 'val', 'test'):
        p = folder / (split+'_manifest.csv')
        df = pd.read_csv(p)
        needed = {'key','label','view','path'}
        if not needed.issubset(df.columns):
            raise ValueError(f'{p}: missing {needed-set(df.columns)}')
        df['split'] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    rawid = full['key'].astype(str).str.split(':').str[-1]
    if (pd.DataFrame({'id':rawid,'split':full['split']}).groupby('id')['split'].nunique()>1).any():
        raise ValueError('Patient overlap across splits')
    if (pd.DataFrame({'id':rawid,'label':full['label']}).groupby('id')['label'].nunique()>1).any():
        raise ValueError('Conflicting patient labels')
    full['patient_id'] = rawid
    full['image_id'] = [hashlib.sha256(str(p).encode()).hexdigest()[:20] for p in full['path']]
    if full['image_id'].duplicated().any():
        raise ValueError('Duplicate image paths in manifests')
    return full


def make_sheets(df, out, score, n=80):
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(df), n):
        chunk = df.iloc[start:start+n]
        tile = 154
        sheet = Image.new('RGB', (tile*8, tile*((len(chunk)+7)//8)), 'white')
        d = ImageDraw.Draw(sheet)
        for j, (_, row) in enumerate(chunk.iterrows()):
            im = Image.open(row['png_path']).convert('RGB').resize((144, 124))
            x, y = (j%8)*tile, (j//8)*tile
            sheet.paste(im, (x,y+23))
            d.text((x+2,y+2), f"{row['image_id'][:9]} {row[score]:.2f}", fill='black')
        sheet.save(out / f'{score}_{start//n+1:03d}.jpg', quality=88)


def prepare(args):
    import joblib
    import tensorflow as tf
    import sklearn
    folder = Path(args.manifests)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = load_manifests(folder)
    # Keep test filenames for integrity checks, but never decode test pixels.
    df[df.split=='test'].to_csv(out/'test_inventory_only.csv', index=False)
    df = df[df.split.isin(['train','val'])].copy()
    if args.smoke:
        df = df.groupby('split', group_keys=False).head(3)
    assets = Path(args.detector)
    for name in ('side_clf.joblib', 'invert_template.npy'):
        if not (assets/name).is_file():
            raise FileNotFoundError(f'Missing teammate asset: {assets/name}')
    print('scikit-learn version:', sklearn.__version__, flush=True)
    if args.sklearn_version and sklearn.__version__ != args.sklearn_version:
        raise RuntimeError(f'scikit-learn mismatch: installed {sklearn.__version__}, expected {args.sklearn_version}')
    clf = joblib.load(assets/'side_clf.joblib')  # only load trusted teammate files
    t = np.load(assets/'invert_template.npy', allow_pickle=False)
    if t.shape != (64,64):
        raise ValueError('Unexpected inversion template shape')
    image_dir = out/'images'
    image_dir.mkdir(exist_ok=True)
    rows = []
    for i, r in enumerate(df.itertuples(index=False),1):
        p = Path(r.path)
        if not p.is_file():
            raise FileNotFoundError(p)
        dest = image_dir / (r.image_id+'.png')
        Image.fromarray(resize224(robust(read_image(p)))).save(dest)
        rows.append({'image_id':r.image_id,'key':r.key,'patient_id':r.patient_id,'label':r.label,'split':r.split,
                     'view':r.view,'orig_path':str(p),'png_path':str(dest)})
        if i % 50 == 0: print('Converted',i,'/',len(df),flush=True)
    man = pd.DataFrame(rows)
    if man.empty:
        raise RuntimeError('No images to process')
    frac_crop = [crop64(p) for p in man['png_path']]
    man['content_frac'] = [x[0] for x in frac_crop]
    man['template_corr'] = [(x[1]*t).mean() for x in frac_crop]
    man['invert'] = man['template_corr'] < 0
    base = tf.keras.applications.MobileNetV2(include_top=False, pooling='avg', input_shape=(224,224,3), weights='imagenet')
    def embed(paths, invert):
        def load(p, inv):
            x = tf.cast(tf.io.decode_png(tf.io.read_file(p),channels=3),tf.float32)
            x = tf.where(inv, 255.0-x, x)
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)
        ds = tf.data.Dataset.from_tensor_slices((list(paths),np.asarray(invert,bool))).map(load).batch(32)
        return base.predict(ds, verbose=0)
    emb = embed(man['png_path'], man['invert'])
    probs = clf.predict_proba(emb)
    if 1 not in list(clf.classes_):
        raise ValueError('Side classifier lacks class 1')
    man['p_side'] = probs[:,list(clf.classes_).index(1)]
    man['drop_side'] = man['p_side'] > 0.5
    man['review_required'] = man['p_side'].between(0.2,0.8) | man['invert'] | (man['content_frac']<0.6)
    man['review_status'] = 'pending'
    man.to_csv(out/'side_invert_flags.csv', index=False)
    make_sheets(man[man.p_side.between(.2,.8)],out/'review_sheets'/'uncertain_side','p_side')
    make_sheets(man[man.invert],out/'review_sheets'/'inverted','template_corr')
    make_sheets(man[man.content_frac<.6],out/'review_sheets'/'large_padding','content_frac')
    # A starter override CSV: edit only these explicit columns, preserving image_id.
    pd.DataFrame({'image_id':man['image_id'],'drop_manual':False,'not_side':False,'not_inverted':False,'reviewed':False,'notes':''}).to_csv(out/'manual_overrides.csv',index=False)
    summary = {'images_scored':len(man),'side_flagged':int(man.drop_side.sum()),'inverted_flagged':int(man.invert.sum()),
               'uncertain_side':int(man.p_side.between(.2,.8).sum()),'large_padding':int((man.content_frac<.6).sum()),
               'requires_review':int(man.review_required.sum()),'test_pixels_loaded':False,'manual_review_completed':False}
    (out/'prepare_summary.json').write_text(json.dumps(summary,indent=2))
    print('PREPARE DONE',summary,flush=True)


def finalize(args):
    out = Path(args.output)
    man = pd.read_csv(out/'side_invert_flags.csv')
    ov = pd.read_csv(out/'manual_overrides.csv').fillna({'notes':''})
    if ov['image_id'].duplicated().any() or set(man.image_id)!=set(ov.image_id):
        raise ValueError('Overrides do not match inventory exactly')
    for col in ('drop_manual','not_side','not_inverted','reviewed'):
        ov[col] = ov[col].astype(str).str.lower().map({'true':True,'false':False})
        if ov[col].isna().any(): raise ValueError(f'Invalid boolean in {col}')
    man = man.merge(ov,on='image_id',validate='one_to_one')
    required = man.review_required.astype(bool)
    if (required & ~man.reviewed).any() and not args.allow_pending:
        raise RuntimeError(f'{int((required & ~man.reviewed).sum())} flagged images require review; edit manual_overrides.csv or pass --allow-pending for exploratory output')
    man['invert_final'] = man['invert'].astype(bool) & ~man['not_inverted']
    man['drop_side_final'] = man['drop_side'].astype(bool) & ~man['not_side']
    man['keep'] = ~(man['drop_side_final'] | man['drop_manual'])
    # Rebuild every retained image from original pixels: inversion before padding.
    for i,r in enumerate(man[man.keep].itertuples(index=False),1):
        a = robust(read_image(r.orig_path))
        if r.invert_final: a = 255-a
        Image.fromarray(resize224(a)).save(r.png_path)
        if i%100==0: print('Finalized',i,flush=True)
    man.to_csv(out/'cleaning_decisions.csv',index=False)
    clean=man[man.keep].copy()
    clean.to_csv(out/'clean_manifest.csv',index=False)
    summary={'images_kept':len(clean),'images_dropped':int((~man.keep).sum()),'patients_kept':int(clean.key.nunique()),
             'review_pending':int((required & ~man.reviewed).sum()),'test_pixels_loaded':False}
    (out/'final_summary.json').write_text(json.dumps(summary,indent=2))
    print('FINALIZE DONE',summary,flush=True)


def features(args):
    import tensorflow as tf
    from tensorflow import keras
    out=Path(args.output)
    df=pd.read_csv(out/'clean_manifest.csv')
    if (df['split']=='test').any(): raise RuntimeError('Test data not permitted')
    model=keras.models.load_model(args.model,compile=False)
    enc=keras.Model(inputs=model.inputs,outputs=model.get_layer('dense').output)
    def load(p):
        x=tf.io.decode_png(tf.io.read_file(p),channels=1)
        return tf.image.grayscale_to_rgb(tf.cast(x,tf.float32)/255.)
    ds=tf.data.Dataset.from_tensor_slices(df['png_path'].tolist()).map(load).batch(32)
    feats=enc.predict(ds,verbose=1)
    if feats.ndim!=2 or feats.shape[1]!=128: raise ValueError(f'Unexpected feature shape: {feats.shape}')
    np.save(out/'modelC_features.npy',feats)
    df[['image_id','key','patient_id','label','split','png_path']].to_csv(out/'modelC_features_index.csv',index=False)
    print('FEATURES DONE',feats.shape,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','finalize','features'])
    p.add_argument('--manifests',default='output/xray_overnight/splits')
    p.add_argument('--output',default='output/xray_modelc_cleaned_v2')
    p.add_argument('--detector',default='xraydata/side_detector')
    p.add_argument('--model',default='xraydata/Models/modelC_resnet50_18oct.keras')
    p.add_argument('--sklearn-version',default=None)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--allow-pending',action='store_true')
    a=p.parse_args()
    {'prepare':prepare,'finalize':finalize,'features':features}[a.stage](a)

if __name__=='__main__':main()
