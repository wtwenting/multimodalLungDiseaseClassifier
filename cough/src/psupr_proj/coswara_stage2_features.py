#!/usr/bin/env python3
"""Coswara Stage 2: train-only acoustic feature extraction and documented analysis."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

FEATURES = ['rms','zcr','centroid_hz','bandwidth_hz','rolloff_hz']
SR=16000
FFT=1024
HOP=256
MELS=64
MFCC=13

def stats(x):
    x=np.asarray(x,dtype=np.float64).ravel()
    return {'mean':float(np.mean(x)),'std':float(np.std(x)),'median':float(np.median(x)), 'p25':float(np.percentile(x,25)), 'p75':float(np.percentile(x,75))}

def calculate(path):
    y,sr=librosa.load(path,sr=None,mono=True)
    if not len(y) or not np.isfinite(y).all() or not np.any(y):
        raise ValueError('empty, silent or nonfinite waveform')
    original_duration=len(y)/sr
    y,_=librosa.effects.trim(y,top_db=60)
    if not len(y) or not np.any(y):
        raise ValueError('empty after trimming')
    if sr!=SR:
        y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    y=y.astype(np.float32)
    # Exclude padded audio from descriptive feature summaries.
    mel=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=FFT,hop_length=HOP,n_mels=MELS)
    logmel=librosa.power_to_db(mel,ref=np.max)
    delta=librosa.feature.delta(logmel)
    delta2=librosa.feature.delta(logmel,order=2)
    mfcc=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=MFCC,n_fft=FFT,hop_length=HOP)
    arrays={
        'rms':librosa.feature.rms(y=y,frame_length=FFT,hop_length=HOP)[0],
        'zcr':librosa.feature.zero_crossing_rate(y,frame_length=FFT,hop_length=HOP)[0],
        'centroid_hz':librosa.feature.spectral_centroid(y=y,sr=SR,n_fft=FFT,hop_length=HOP)[0],
        'bandwidth_hz':librosa.feature.spectral_bandwidth(y=y,sr=SR,n_fft=FFT,hop_length=HOP)[0],
        'rolloff_hz':librosa.feature.spectral_rolloff(y=y,sr=SR,n_fft=FFT,hop_length=HOP)[0],
        'logmel':logmel,'delta':delta,'delta2':delta2,
    }
    for i in range(MFCC): arrays[f'mfcc_{i+1:02d}']=mfcc[i]
    out={'original_duration_sec':float(original_duration),'trimmed_duration_sec':len(y)/SR,'original_sr':int(sr)}
    for name,a in arrays.items():
        for stat,value in stats(a).items():out[f'{name}_{stat}']=value
    return out

def run(args):
    out=Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    with open(args.metadata,newline='',encoding='utf-8') as f: records=list(csv.DictReader(f))
    train=[r for r in records if r.get('split','').strip()=='train' and r.get('status','').strip() in ('healthy','COVID-19')]
    if not train:raise ValueError('No train records; use coswara_heavy_split.csv with split column')
    if args.limit:train=train[:args.limit]
    rows=[]; failures=[]
    for i,r in enumerate(train,1):
        try:
            row={'uuid':r['uuid'],'status':r['status'],'split':'train','audio_path':r['audio_path']}
            row.update(calculate(r['audio_path']))
            rows.append(row)
        except Exception as e:failures.append({'uuid':r.get('uuid'),'error':str(e)})
        if i%50==0 or i==len(train):print(f'Processed {i}/{len(train)}; valid={len(rows)}, failed={len(failures)}',flush=True)
    if not rows:raise RuntimeError('No audio successfully processed. Check paths in metadata CSV.')
    features_path=out/'coswara_acoustic_features.csv'
    with features_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    numeric=[k for k in rows[0] if k not in ('uuid','status','split','audio_path','original_sr')]
    summaries=[];separability=[]
    for name in numeric:
        groups={label:np.array([float(r[name]) for r in rows if r['status']==label]) for label in ('healthy','COVID-19')}
        if any(len(v)==0 for v in groups.values()):continue
        for label,v in groups.items():
            summaries.append({'feature':name,'class':label,'count':len(v),**stats(v)})
        a,b=groups['healthy'],groups['COVID-19']
        pooled=np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2)) if min(len(a),len(b))>1 else 0
        d=(np.mean(b)-np.mean(a))/pooled if pooled>0 else 0.0
        separability.append({'feature':name,'healthy_mean':float(np.mean(a)),'covid_mean':float(np.mean(b)),'cohens_d_covid_minus_healthy':float(d),'abs_cohens_d':abs(float(d))})
    for filename,data in [('coswara_feature_summary.csv',summaries),('coswara_feature_separability.csv',separability)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    config={'metadata':str(Path(args.metadata).resolve()),'scope':'train split only','audio_type':'Coswara heavy cough','sample_rate_hz':SR,'trim_top_db':60,'n_fft':FFT,'hop_length':HOP,'n_mels':MELS,'n_mfcc':MFCC,'feature_duration':'trimmed audio before zero padding','delta':'librosa.feature.delta(logmel)','delta_delta':'librosa.feature.delta(logmel,order=2)','recording_summary':['mean','std','median','p25','p75'],'randomness':'none'}
    (out/'coswara_feature_config.json').write_text(json.dumps(config,indent=2)+'\n')
    validation={'expected_train_records':len(train),'processed':len(rows),'failed':len(failures),'class_counts':dict(Counter(r['status'] for r in rows)),'numeric_feature_columns':len(numeric),'nonfinite_values':sum(not math.isfinite(float(r[k])) for r in rows for k in numeric),'failures':failures}
    (out/'coswara_feature_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    # Visuals use a headless backend and are optional.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=out/'plots';plots.mkdir(exist_ok=True)
    plot_features=['rms_mean','zcr_mean','centroid_hz_mean','bandwidth_hz_mean','rolloff_hz_mean','mfcc_01_mean','logmel_std','delta_std','delta2_std']
    for name in plot_features:
        data=[[float(r[name]) for r in rows if r['status']==label] for label in ('healthy','COVID-19')]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.boxplot(data,tick_labels=['Healthy','COVID-19'],showfliers=False)
        ax.set_title(name.replace('_',' ').title());ax.set_ylabel(name)
        fig.tight_layout();fig.savefig(plots/f'{name}.png',dpi=150);plt.close(fig)
    ranking=sorted(separability,key=lambda r:r['abs_cohens_d'],reverse=True)
    report=['# Coswara Stage 2 — Acoustic Feature Extraction and Analysis','',f'**Scope:** {len(rows)} successfully processed training recordings; {len(failures)} failures.','',
            '## Method','',f'Original mono audio → `librosa.effects.trim(top_db=60)` → resample to {SR} Hz → calculate frame-level features on **unpadded trimmed audio** → summarize each recording.','',
            'Features: RMS, zero-crossing rate, spectral centroid, bandwidth, rolloff, 13 MFCC coefficients, 64-band Log-Mel, first-order Delta and second-order Delta-delta.','',
            'All descriptive summaries use mean, standard deviation, median, and quartiles. Features are extracted only from the training split to prevent test-guided feature selection.','',
            '## Validation','',f'- Processed: {len(rows)} / {len(train)}','- Healthy: '+str(validation['class_counts'].get('healthy',0)),'- COVID-19: '+str(validation['class_counts'].get('COVID-19',0)),f'- Nonfinite feature values: {validation["nonfinite_values"]}','',
            '## Largest standardized mean differences (exploratory)','', '| Feature | Cohen d (COVID minus Healthy) |','|---|---:|']
    for r in ranking[:12]:report.append(f'| {r["feature"]} | {r["cohens_d_covid_minus_healthy"]:+.3f} |')
    report += ['','**Interpretation caution:** A large univariate effect size does not establish diagnostic validity or improve CNN Macro F1. Overlapping features may still be useful jointly; effects may reflect microphone, environment, age, or recording-duration confounders.','',
               '## Stage 3 handoff','', 'Test Log-Mel baseline versus Log-Mel + Delta + Delta-delta using identical participant splits and validation-based model selection. Address class imbalance and tune decision thresholds using validation data only. Do not choose hyperparameters using the test set.','',
               '## Output files','', '- `coswara_acoustic_features.csv`: one row per training recording','- `coswara_feature_summary.csv`: class-wise statistics','- `coswara_feature_separability.csv`: exploratory standardized mean differences','- `coswara_feature_validation.json`: audit and failures','- `coswara_feature_config.json`: reproducibility parameters','- `plots/`: boxplots','']
    (out/'stage2_feature_report.md').write_text('\n'.join(report),encoding='utf-8')
    print('Saved Stage 2 outputs to:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',required=True,help='Path to coswara_heavy_split.csv')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--limit',type=int,default=None,help='Optional smoke-test limit')
    run(p.parse_args())
