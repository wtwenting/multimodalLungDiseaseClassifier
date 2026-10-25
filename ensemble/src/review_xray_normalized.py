#!/usr/bin/env python3
"""Label-blind review sheets for flagged test X-rays; no inference or QC decisions."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_modelc_xray import read_image, robust, direct


def panel(arr, width=290, height=290):
    im = Image.fromarray(arr.astype(np.uint8), mode='L')
    im.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), '#eeeeee')
    canvas.paste(im.convert('RGB'), ((width-im.width)//2, (height-im.height)//2))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qc', type=Path, default=Path('output/xray_test_features/test_image_qc.csv'))
    p.add_argument('--out', type=Path, default=Path('output/xray_test_features/manual_review'))
    args = p.parse_args()
    df = pd.read_csv(args.qc)
    required = {'patient_id', 'source_path', 'status', 'inversion_detected'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
    flagged = df.loc[df.status.astype(str).eq('REVIEW')].copy().reset_index(drop=True)
    if flagged.empty:
        raise ValueError('No REVIEW images in QC file')
    args.out.mkdir(parents=True, exist_ok=True)
    w, h, gap, header = 290, 290, 12, 45
    tile_w, tile_h = 2*w+3*gap, h+header+2*gap
    sheet = Image.new('RGB', (tile_w*2, tile_h*((len(flagged)+1)//2)), 'white')
    draw = ImageDraw.Draw(sheet)
    rows = []
    for i, r in flagged.iterrows():
        source = Path(str(r.source_path))
        if not source.is_file():
            raise FileNotFoundError(source)
        raw = read_image(source)
        # Original training Model C uses robust 1st-99th percentile normalization.
        normalized = robust(raw)
        inv = str(r.inversion_detected).strip().lower() in ('true', '1', 'yes')
        corrected = (255-normalized) if inv else normalized
        col, row = i%2, i//2
        x0, y0 = col*tile_w, row*tile_h
        draw.text((x0+gap, y0+8), f'IMG_{i+1:02d}  Patient {r.patient_id}  inversion_flag={inv}', fill='black')
        sheet.paste(panel(normalized, w, h), (x0+gap, y0+header))
        sheet.paste(panel(corrected, w, h), (x0+2*gap+w, y0+header))
        draw.text((x0+gap, y0+header+h+2), 'Robust 1-99%', fill='black')
        draw.text((x0+2*gap+w, y0+header+h+2), 'Flag-corrected (if flagged)', fill='black')
        rows.append({'review_id':f'IMG_{i+1:02d}', 'patient_id':str(r.patient_id),
                     'source_path':str(source), 'inversion_detected':inv,
                     'manual_decision':'UNRESOLVED', 'review_notes':''})
    dest = args.out/'normalized_review_contact_sheet.png'
    sheet.save(dest)
    csv = args.out/'normalized_review_decisions.csv'
    # Never overwrite completed review decisions.
    if not csv.exists():
        pd.DataFrame(rows).to_csv(csv, index=False)
    print(f'Review images: {len(rows)}; patients: {flagged.patient_id.nunique()}')
    print(f'Contact sheet: {dest}')
    print(f'Decisions template: {csv}')
    print('Preview only. No labels, predictions, feature files or original images changed.')


if __name__ == '__main__':
    main()
