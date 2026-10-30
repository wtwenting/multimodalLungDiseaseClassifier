#!/usr/bin/env python3
"""Controlled Model B vs Model C on identical finalized BIMCV X-ray cohort.
No test images are read. Existing files are not modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, precision_score, roc_auc_score,
                             confusion_matrix)

DEFAULT_B = '/Users/wenting/Projects/PsuprProj/xraydata/Models/ModelB_CovidvsNormal_27sep/model_B_v2_27sep.keras'

def read_image(path):
    with Image.open(path) as im:
        a = np.asarray(im)
        if a.ndim == 3:
            a = np.asarray(im.convert('L'))
    if a.ndim != 2:
        raise ValueError(f'Not grayscale: {path}')
    return a.astype(np.float32)

def direct(a):
    if a.max() > 255:
        a = np.floor(a / 256.0)
    return np.clip(a, 0, 255).astype(np.uint8)

def robust(a):
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return direct(a)
    return np.clip((a-lo)*255/(hi-lo), 0, 255).astype(np.uint8)

def model_b_pixels(row, decision, cv2):
    a = read_image(row.orig_path)
    rule = decision['selective_clahe_decision']
    if rule == 'direct':
        a8 = direct(a)
    elif rule == 'robust_1_99':
        a8 = robust(a)
    elif rule == 'clahe_after_robust':
        a8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(robust(a))
    else:
        raise ValueError(f'Unknown adaptive rule {rule}')
    # The cleaning correction is applied before padding; do not invert black padding.
    if str(row.invert_final).lower() == 'true':
        a8 = 255-a8
    return a8

def metrics(y, p, threshold):
    z = (p >= threshold).astype(int)
    return dict(threshold=round(float(threshold), 6),
                accuracy=float(accuracy_score(y,z)),
                balanced_accuracy=float(balanced_accuracy_score(y,z)),
                macro_f1=float(f1_score(y,z,average='macro',zero_division=0)),
                covid_recall=float(recall_score(y,z,pos_label=1,zero_division=0)),
                noncovid_recall=float(recall_score(y,z,pos_label=0,zero_division=0)),
                covid_precision=float(precision_score(y,z,pos_label=1,zero_division=0)),
                roc_auc=float(roc_auc_score(y,p)),
                confusion_matrix=confusion_matrix(y,z,labels=[0,1]).tolist())

def evaluate(name, X, df, out, C, fixed_threshold):
    tr = df.split.eq('train').to_numpy()
    va = df.split.eq('val').to_numpy()
    counts = df.loc[tr].groupby('key')['image_id'].transform('size').to_numpy()
    weights = 1.0 / counts
    weights *= len(weights)/weights.sum()
    model = make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=42))
    model.fit(X[tr],df.loc[tr,'label'].to_numpy(),logisticregression__sample_weight=weights)
    val = df.loc[va,['key','label','image_id']].copy()
    val['probability'] = model.predict_proba(X[va])[:,1]
    val.to_csv(out / f'{name}_image_predictions.csv',index=False)
    patients = val.groupby('key',sort=True).agg(label=('label','first'),probability=('probability','mean'),images=('image_id','size')).reset_index()
    patients.to_csv(out / f'{name}_patient_predictions.csv',index=False)
    y = patients.label.to_numpy(); p = patients.probability.to_numpy()
    fixed = metrics(y,p,fixed_threshold)
    tuned = max((metrics(y,p,t) for t in np.arange(0.2,0.80001,0.025)),
                key=lambda m:(m['macro_f1'],m['balanced_accuracy'],-abs(m['threshold']-.5)))
    print(name,'FIXED',json.dumps(fixed),flush=True)
    print(name,'TUNED exploratory',json.dumps(tuned),flush=True)
    return {'encoder':name,'fixed':fixed,'tuned_exploratory':tuned,'train_images':int(tr.sum()),'val_images':int(va.sum()),'val_patients':len(patients)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clean',type=Path,default=Path('output/xray_modelc_cleaned_v2'))
    ap.add_argument('--audit',type=Path,default=Path('output/xray_adaptive_audit'))
    ap.add_argument('--model-b',type=Path,default=Path(DEFAULT_B))
    ap.add_argument('--output',type=Path,default=Path('output/xray_modelb_vs_modelc_cleaned'))
    ap.add_argument('--c',type=float,default=.01)
    ap.add_argument('--threshold',type=float,default=.425)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((a.clean/'final_summary.json').read_text())
    if summary.get('review_pending',-1)!=0: raise RuntimeError('Pending manual review')
    df=pd.read_csv(a.clean/'clean_manifest.csv')
    idx=pd.read_csv(a.clean/'modelC_features_index.csv')
    Xc=np.load(a.clean/'modelC_features.npy',allow_pickle=False)
    needed=['image_id','key','patient_id','label','split','png_path']
    if not df[needed].reset_index(drop=True).equals(idx[needed].reset_index(drop=True)):
        raise RuntimeError('Model C feature index does not exactly match cleaned manifest order')
    if Xc.shape!=(len(df),128) or not np.isfinite(Xc).all(): raise RuntimeError('Bad Model C features')
    if set(df.split)!={'train','val'}: raise RuntimeError('Unexpected split, test is forbidden')
    if df.image_id.duplicated().any(): raise RuntimeError('Duplicate image IDs')
    if df.groupby('patient_id').split.nunique().max()!=1: raise RuntimeError('Patient split leakage')
    if df.groupby('patient_id').label.nunique().max()!=1: raise RuntimeError('Patient label conflict')
    if df.groupby('key').label.nunique().max()!=1: raise RuntimeError('Key label conflict')
    if not a.model_b.is_file(): raise FileNotFoundError(a.model_b)
    audit=pd.read_csv(a.audit/'preprocessing_decisions.csv')
    if audit.path.duplicated().any(): raise RuntimeError('Duplicate audit paths')
    decisions=audit.set_index('path').to_dict('index')
    for r in df.itertuples(index=False):
        if r.orig_path not in decisions: raise RuntimeError(f'Missing audit decision {r.orig_path}')
        if decisions[r.orig_path]['key']!=r.key or decisions[r.orig_path]['split']!=r.split:
            raise RuntimeError(f'Wrong audit decision for {r.orig_path}')
    print('Cohort validated:',len(df),'images,',df.key.nunique(),'patients;',df.split.value_counts().to_dict(),flush=True)
    print('TEST NOT LOADED OR EVALUATED',flush=True)
    if a.smoke:
        print('SMOKE PASS: cohort, Model C features, audit rules and Model B path verified',flush=True)
        return
    import tensorflow as tf
    import cv2
    full=tf.keras.models.load_model(str(a.model_b),compile=False)
    if tuple(full.input_shape[1:])!=(224,224,3): raise RuntimeError('Wrong Model B input')
    enc=tf.keras.Model(full.inputs,full.get_layer('dense').output)
    enc.trainable=False
    # Cache is keyed by model bytes, cleaned cohort, inversion decisions and audit rules.
    hasher=hashlib.sha256()
    with a.model_b.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            hasher.update(chunk)
    modelhash=hasher.hexdigest()
    tag=hashlib.sha256(json.dumps({'model':modelhash,'rows':[(str(r.image_id),str(r.orig_path),str(r.invert_final),str(decisions[r.orig_path]['selective_clahe_decision'])) for r in df.itertuples(index=False)],'preprocessing':'B selective CLAHE + finalized inversion correction; TF resize_with_pad 224; 0-1'},sort_keys=True).encode()).hexdigest()[:16]
    cache_dir=a.output/'features'/tag
    cache_dir.mkdir(parents=True,exist_ok=True)
    Xb=[]
    for n,r in enumerate(df.itertuples(index=False),1):
        cache=cache_dir/(str(r.image_id)+'.npy')
        if cache.is_file():
            vec=np.load(cache,allow_pickle=False)
        else:
            arr=model_b_pixels(r,decisions[r.orig_path],cv2)
            x=tf.image.resize_with_pad(tf.convert_to_tensor(arr[...,None],tf.float32),224,224,antialias=False)
            x=tf.repeat(x,3,axis=-1)/255.0
            vec=np.asarray(enc(x[None,...],training=False)).reshape(-1).astype(np.float32)
            np.save(cache,vec)
        if vec.shape!=(128,) or not np.isfinite(vec).all(): raise RuntimeError(f'Invalid B features {r.image_id}')
        Xb.append(vec)
        if n%50==0 or n==len(df): print('Model B features',n,'/',len(df),flush=True)
    Xb=np.stack(Xb)
    results=[evaluate('model_b_selective_clahe',Xb,df,a.output,a.c,a.threshold),evaluate('model_c_robust_cleaned',Xc,df,a.output,a.c,a.threshold)]
    records=[]
    for r in results:
        for mode in ('fixed','tuned_exploratory'):
            records.append({'encoder':r['encoder'],'mode':mode,'train_images':r['train_images'],'val_images':r['val_images'],'val_patients':r['val_patients'],**{k:v for k,v in r[mode].items() if k!='confusion_matrix'},'confusion_matrix':json.dumps(r[mode]['confusion_matrix'])})
    pd.DataFrame(records).to_csv(a.output/'validation_comparison.csv',index=False)
    report={'same_cohort':True,'cohort_images':len(df),'cohort_patients':int(df.key.nunique()),'model_b_sha256':modelhash,'model_b_cache_tag':tag,'model_c_feature_source':str(a.clean/'modelC_features.npy'),'classification':'Patient-balanced logistic regression C=0.01; patient mean probability','results':results,'test':'NOT_LOADED_OR_EVALUATED','caveats':['Model B uses selective CLAHE from prior audit plus finalized inversion correction; this differs from historical Model B experiment.','Both models use identical retained images and patient assignments; encoder-specific preprocessing differs by design.','Threshold tuning on validation is exploratory, not independent test performance.']}
    (a.output/'experiment_summary.json').write_text(json.dumps(report,indent=2))
    print('DONE',a.output.resolve(),flush=True)

if __name__=='__main__':main()
