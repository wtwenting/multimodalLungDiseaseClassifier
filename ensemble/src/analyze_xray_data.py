#!/usr/bin/env python3
"""Read-only train/validation diagnostics for cleaned BIMCV X-ray MLP."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clean',type=Path,default=Path('output/xray_modelc_cleaned_v2/clean_manifest.csv'))
    ap.add_argument('--pred',type=Path,default=Path('output/xray_mlp_256d/validation_patient_predictions.csv'))
    ap.add_argument('--output',type=Path,default=Path('output/xray_data_diagnostics'))
    ap.add_argument('--threshold',type=float,default=0.425)
    args=ap.parse_args()
    df=pd.read_csv(args.clean)
    required={'key','patient_id','label','split','image_id'}
    if not required.issubset(df.columns):raise ValueError('Clean manifest missing columns: '+str(required-set(df.columns)))
    if not set(df.split.unique()).issubset({'train','val'}):raise ValueError('Unexpected split, refusing test data')
    if df.image_id.duplicated().any():raise ValueError('Duplicate image IDs')
    if (df.groupby('patient_id').split.nunique()>1).any():raise ValueError('Patient overlap')
    if (df.groupby('patient_id').label.nunique()>1).any():raise ValueError('Conflicting labels')
    pats=df.groupby(['split','key','patient_id','label'],as_index=False).agg(image_count=('image_id','size'))
    pats['class_name']=pats.label.map({0:'COVID-negative',1:'COVID-positive'})
    summary=pats.groupby(['split','class_name'],as_index=False).agg(patients=('key','size'),images=('image_count','sum'),mean_images_per_patient=('image_count','mean'),median_images_per_patient=('image_count','median'),max_images_per_patient=('image_count','max'))
    args.output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output/'class_distribution.csv',index=False)
    pats.to_csv(args.output/'images_per_patient.csv',index=False)
    pred=pd.read_csv(args.pred)
    if not {'key','label','probability'}.issubset(pred.columns):raise ValueError('Predictions missing key/label/probability')
    if pred.key.duplicated().any():raise ValueError('Predictions must be one row per patient')
    vp=pats[pats.split.eq('val')].set_index('key')
    pp=pred.set_index('key')
    if set(pp.index)!=set(vp.index):raise ValueError('Prediction patient IDs do not match validation cohort')
    pp=pp.loc[vp.index].copy()
    if not np.array_equal(pp.label.to_numpy(),vp.label.to_numpy()):raise ValueError('Prediction labels mismatch')
    if not np.isfinite(pp.probability).all() or not pp.probability.between(0,1).all():raise ValueError('Invalid probabilities')
    pp['class_name']=pp.label.map({0:'COVID-negative',1:'COVID-positive'})
    pp['predicted_label']=(pp.probability>=args.threshold).astype(int)
    pp['outcome']=np.select([(pp.label==1)&(pp.predicted_label==1),(pp.label==0)&(pp.predicted_label==0),(pp.label==0)&(pp.predicted_label==1),(pp.label==1)&(pp.predicted_label==0)],['TP','TN','FP','FN'],default='?')
    pp['image_count']=vp.image_count
    pp.reset_index().to_csv(args.output/'validation_prediction_audit.csv',index=False)
    probs=pp.groupby('class_name').probability.agg(['count','mean','median','std','min','max']).reset_index()
    probs.to_csv(args.output/'probability_by_class.csv',index=False)
    bins=np.linspace(0,1,11)
    hist=[]
    for name,group in pp.groupby('class_name'):
        counts,_=np.histogram(group.probability,bins=bins)
        for i,n in enumerate(counts):hist.append({'class_name':name,'bin_start':round(float(bins[i]),2),'bin_end':round(float(bins[i+1]),2),'patients':int(n)})
    pd.DataFrame(hist).to_csv(args.output/'probability_histogram.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,group in pp.groupby('class_name'):
        ax.hist(group.probability,bins=bins,alpha=.55,label=name)
    ax.axvline(args.threshold,linestyle='--',color='black',label=f'Threshold {args.threshold:.3f}')
    ax.set(xlabel='Predicted COVID probability',ylabel='Number of patients',title='Validation predicted probabilities by actual class')
    ax.legend();fig.tight_layout();fig.savefig(args.output/'probability_histogram.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    tr=pats[pats.split.eq('train')]
    for name,group in tr.groupby('class_name'):
        ax.hist(group.image_count,bins=np.arange(.5,tr.image_count.max()+1.5,1),alpha=.55,label=name)
    ax.set(xlabel='Images per training patient',ylabel='Patients',title='Training images per patient')
    ax.legend();fig.tight_layout();fig.savefig(args.output/'images_per_patient.png',dpi=160);plt.close(fig)
    report={'train_patients':int(sum(pats.split=='train')),'val_patients':int(sum(pats.split=='val')),'train_images':int(sum(df.split=='train')),'val_images':int(sum(df.split=='val')),'threshold':args.threshold,'outcomes':{str(k):int(v) for k,v in pp.outcome.value_counts().items()},'test':'NOT_LOADED'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print('CLASS DISTRIBUTION\n'+summary.to_string(index=False))
    print('\nPROBABILITIES\n'+probs.to_string(index=False))
    print('\nOUTCOMES',report['outcomes'])
    print('DONE',args.output.resolve())

if __name__=='__main__':main()
