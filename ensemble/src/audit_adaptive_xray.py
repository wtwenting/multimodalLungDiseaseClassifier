#!/usr/bin/env python3
"""Read-only, split-aware BIMCV X-ray intensity audit and adaptive preprocessing preview."""
import argparse, csv, json, hashlib, math, sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps, ImageDraw

METHODS = ('direct', 'adaptive', 'selective_clahe')

def read_gray(path):
    with Image.open(path) as im:
        a = np.asarray(im)
        mode = im.mode
    if a.ndim == 3:
        a = np.asarray(Image.fromarray(a).convert('L'))
    if a.ndim != 2 or a.size == 0:
        raise ValueError('Not a 2D X-ray')
    return a.astype(np.float32), mode

def direct(a):
    if a.max() > 255:
        a = np.floor(a / 256.0)
    return np.clip(a, 0, 255).astype(np.uint8)

def robust(a):
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return direct(a)
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

def clahe(a):
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError('Install opencv-python to use selective CLAHE') from e
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a)

def cnn_input(a):
    # Keep the aspect ratio, black pad to 224 square, then normalize to [0,1].
    im = Image.fromarray(a, 'L').convert('RGB')
    im.thumbnail((224,224), Image.Resampling.BILINEAR)
    dst = Image.new('RGB', (224,224), (0,0,0))
    dst.paste(im, ((224-im.width)//2, (224-im.height)//2))
    return np.asarray(dst).astype(np.float32)/255.0

def stats(a):
    # Central chest is a rough ROI, not a lung mask. Avoid black borders when deciding.
    h,w = a.shape
    roi = a[int(h*.15):max(int(h*.85),int(h*.15)+1), int(w*.2):max(int(w*.8),int(w*.2)+1)]
    if roi.size == 0: roi = a
    p1,p50,p99 = np.percentile(roi, [1,50,99])
    d = direct(a)
    dr = d[int(h*.15):max(int(h*.85),int(h*.15)+1), int(w*.2):max(int(w*.8),int(w*.2)+1)]
    return {'raw_min':float(a.min()), 'raw_max':float(a.max()),'roi_p1':float(p1),
            'roi_median':float(p50),'roi_p99':float(p99),'roi_range':float(p99-p1),
            'direct_roi_mean':float(dr.mean()),'direct_roi_std':float(dr.std()),
            'direct_roi_dark_pct':float(np.mean(dr<=10)*100),
            'direct_roi_bright_pct':float(np.mean(dr>=245)*100)}

def pct(x): return float(np.percentile(x,[5,95])[0]),float(np.percentile(x,[5,95])[1])

def classify(s, thresholds):
    # All thresholds derived from TRAIN patients only; no labels used in routing.
    if s['direct_roi_mean'] < thresholds['mean_low'] or s['direct_roi_dark_pct'] > thresholds['dark_high']:
        return 'dark'
    if s['direct_roi_mean'] > thresholds['mean_high']:
        return 'bright'
    if s['direct_roi_std'] < thresholds['std_low']:
        return 'low_contrast'
    return 'normal'

def apply_method(a, quality, method):
    if method == 'direct': return direct(a)
    if method == 'adaptive': return robust(a) if quality in ('dark','bright') else direct(a)
    if method == 'selective_clahe':
        return clahe(robust(a)) if quality == 'low_contrast' else (robust(a) if quality in ('dark','bright') else direct(a))
    raise ValueError(method)

def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits',type=Path,default=Path('output/xray_overnight/splits'))
    ap.add_argument('--output',type=Path,default=Path('output/xray_adaptive_audit'))
    ap.add_argument('--smoke',action='store_true',help='Audit up to 12 training images')
    ap.add_argument('--preview-per-group',type=int,default=3)
    ap.add_argument('--skip-clahe',action='store_true')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ('train','val','test'):
        path=args.splits/f'{split}_manifest.csv'
        if not path.exists(): ap.error(f'Missing split manifest: {path}')
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                if row.get('split')!=split: raise ValueError(f'Split mismatch: {row}')
                if row.get('view') not in ('AP','PA'): continue
                row['split']=split
                rows.append(row)
    if args.smoke: rows=[r for r in rows if r['split']=='train'][:12]
    print(f'Inventory: {len(rows)} images; smoke={args.smoke}',flush=True)
    records=[]; failures=[]
    for i,row in enumerate(rows,1):
        path=Path(row['path'])
        try:
            st=path.stat()
            if int(row['bytes'])!=st.st_size or int(row['mtime_ns'])!=st.st_mtime_ns:
                raise ValueError('File changed since split manifest was generated')
            a,mode=read_gray(path)
            s=stats(a)
            records.append({'key':row['key'],'label':row['label'],'split':row['split'],
                            'view':row['view'],'path':str(path),'mode':mode,
                            'height':a.shape[0],'width':a.shape[1],**s})
        except Exception as e:
            failures.append({'path':str(path),'error':str(e)})
        if i%100==0 or i==len(rows):print(f'Analysed {i}/{len(rows)} | failures={len(failures)}',flush=True)
    if failures:
        write_csv(args.output/'failures.csv',failures)
        raise RuntimeError(f'{len(failures)} files failed validation; review failures.csv; no decisions generated')
    train=[r for r in records if r['split']=='train']
    if not train: raise RuntimeError('No TRAIN images available')
    # Fit thresholds at patient level so frequent-image patients have equal influence.
    by_patient={}
    for r in train:by_patient.setdefault(r['key'],[]).append(r)
    patient_stats=[]
    for group in by_patient.values():
        patient_stats.append({k:float(np.median([r[k] for r in group])) for k in ('direct_roi_mean','direct_roi_std','direct_roi_dark_pct')})
    def values(k):return np.array([p[k] for p in patient_stats])
    thresholds={
        'mean_low':float(np.percentile(values('direct_roi_mean'),10)),
        'mean_high':float(np.percentile(values('direct_roi_mean'),90)),
        'std_low':float(np.percentile(values('direct_roi_std'),10)),
        'dark_high':float(np.percentile(values('direct_roi_dark_pct'),90)),
    }
    for r in records:
        r['quality_flag']=classify(r,thresholds)
        r['adaptive_decision']='robust_1_99' if r['quality_flag'] in ('dark','bright') else 'direct'
        r['selective_clahe_decision']=('clahe_after_robust' if r['quality_flag']=='low_contrast' else r['adaptive_decision'])
    write_csv(args.output/'image_quality_inventory.csv',records)
    write_csv(args.output/'preprocessing_decisions.csv',[
        {k:r[k] for k in ('key','split','view','path','quality_flag','adaptive_decision','selective_clahe_decision')}
        for r in records])
    (args.output/'thresholds_train_only.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    # Visualize TRAIN only, avoiding accidental test-set inspection.
    selected=[]
    for group in ('dark','bright','low_contrast','normal'):
        selected.extend([r for r in records if r['split']=='train' and r['quality_flag']==group][:args.preview_per_group])
    preview_dir=args.output/'comparisons';preview_dir.mkdir(exist_ok=True)
    methods=['direct','adaptive']+([] if args.skip_clahe else ['selective_clahe'])
    for i,r in enumerate(selected,1):
        a,_=read_gray(Path(r['path']))
        panels=[]
        for method in methods:
            x=cnn_input(apply_method(a,r['quality_flag'],method))
            if x.shape!=(224,224,3) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
                raise RuntimeError(f'Invalid CNN input for {r["path"]} {method}')
            panels.append(Image.fromarray(np.round(x*255).astype('uint8')))
        sheet=Image.new('RGB',(224*len(panels),250),'white')
        draw=ImageDraw.Draw(sheet)
        for j,(name,panel) in enumerate(zip(methods,panels)):
            sheet.paste(panel,(j*224,26));draw.text((j*224+8,7),name,fill='black')
        sheet.save(preview_dir/f'{i:02d}_{r["quality_flag"]}_{r["key"].replace(":","_")}.png')
    summary={'images':len(records),'patients':len({r['key'] for r in records}),
             'images_by_split':dict(Counter(r['split'] for r in records)),
             'patients_by_split':{s:len({r['key'] for r in records if r['split']==s}) for s in ('train','val','test')},
             'quality_flags':dict(Counter(r['quality_flag'] for r in records)),
             'quality_flags_by_split':{s:dict(Counter(r['quality_flag'] for r in records if r['split']==s)) for s in ('train','val','test')},
             'train_patient_count_for_thresholds':len(by_patient),
             'thresholds':thresholds,'preview_images':len(selected),
             'source_images_modified':False,'trained_model':False,
             'warning':'Quality flags are heuristics, not clinical diagnoses. Central ROI is not lung segmentation. No model performance measured. Test labels and pixels are included only for automated inventory, not for threshold fitting or visual preview.'}
    (args.output/'quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE:',json.dumps(summary,sort_keys=True),flush=True)
    print('Outputs:',args.output.resolve(),flush=True)

if __name__=='__main__':
    main()
