#!/usr/bin/env python3
"""Package frozen BIMCV CT+X-ray inference assets for teammate GUI handover.
Run from PsuprProj/ensemble: python src/package_fusion_gui.py --dry-run
Then: python src/package_fusion_gui.py
Only explicitly allowlisted files are packaged. No patient datasets or predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone

FILES = {
    'models/fusion/ct_b_xray_logistic_frozen.joblib': 'ensemble/output/fusion_frozen_final/ct_b_xray_logistic_frozen.joblib',
    'models/ct/v61_13a378082055_best.pt': 'ensemble/models/ct_v61/v61_13a378082055_best.pt',
    'models/xray/model_B_v2_27sep.keras': 'xraydata/Models/ModelB_CovidvsNormal_27sep/model_B_v2_27sep.keras',
    'models/xray/modelC_resnet50_18oct.keras': 'xraydata/Models/modelC_resnet50_18oct.keras',
    'models/side_detector/side_clf.joblib': 'ensemble/models/side_detector/side_clf.joblib',
    'models/side_detector/invert_template.npy': 'ensemble/models/side_detector/invert_template.npy',
    'config/xray_thresholds_train_only.json': 'ensemble/output/xray_adaptive_audit/thresholds_train_only.json',
    'config/ct_preprocessing_config.json': 'ensemble/ct_v5_corrected_preprocessed/preprocessing_config.json',
    'reports/freeze_report.json': 'ensemble/output/fusion_frozen_final/freeze_report.json',
    'reports/final_test_summary.json': 'ensemble/output/fusion_final_test/final_test_summary.json',
    'src/ct_v61_frozen_baseline.py': 'ensemble/src/ct_v61_frozen_baseline.py',
    'src/train_ct_v61.py': 'ensemble/src/train_ct_v61.py',
    'src/prepare_ct_v5_final.py': 'ensemble/src/prepare_ct_v5_final.py',
    'src/clean_modelc_xray.py': 'ensemble/src/clean_modelc_xray.py',
    'src/compare_encoders.py': 'ensemble/src/compare_encoders.py',
    'src/audit_adaptive_xray.py': 'ensemble/src/audit_adaptive_xray.py',
    'src/extract_ct_test_features.py': 'ensemble/src/extract_ct_test_features.py',
    'src/extract_xray_test_features_reviewed.py': 'ensemble/src/extract_xray_test_features_reviewed.py',
    'src/evaluate_fusion_test.py': 'ensemble/src/evaluate_fusion_test.py',
}
# The preprocessing configuration can vary in location; do not silently substitute.
OPTIONAL = {'config/ct_preprocessing_config.json', 'src/train_ct_v61.py', 'src/prepare_ct_v5_final.py', 'reports/final_test_summary.json'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=None, help='PsuprProj root; defaults to parent of ensemble/')
    ap.add_argument('--handover', type=Path, default=None, help='Path to HANDOVER.md; defaults to alongside this script')
    ap.add_argument('--output', type=Path, default=None, help='ZIP output path')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    project = (args.project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    handover = (args.handover or Path(__file__).with_name('HANDOVER.md')).expanduser().resolve()
    output = (args.output or (project / 'ensemble' / 'output' / 'handover' / 'CT_Xray_GUI_Handover.zip')).expanduser().resolve()
    if not (project / 'ensemble').is_dir():
        ap.error(f'Expected PsuprProj/ensemble under {project}; supply --project-root')
    if not handover.is_file():
        ap.error(f'Missing handover: {handover}. Save HANDOVER.md beside this script.')
    missing_required, missing_optional, selected = [], [], []
    for archive_name, rel in FILES.items():
        source = project / rel
        if not source.is_file():
            (missing_optional if archive_name in OPTIONAL else missing_required).append(rel)
        else:
            selected.append((archive_name, source))
    print(f'Project root: {project}')
    print(f'Included assets: {len(selected)} plus HANDOVER.md and manifest.json')
    print(f'Missing optional: {missing_optional}')
    if missing_required:
        print('MISSING REQUIRED:', *missing_required, sep='\n  ', file=sys.stderr)
        sys.exit(2)
    if args.dry_run:
        for name, path in selected:
            print(f'  {name} ({path.stat().st_size / (1024*1024):.2f} MiB)')
        print('DRY RUN PASS; no ZIP created')
        return
    if output == handover or any(output == path.resolve() for _, path in selected):
        ap.error('Output ZIP path would overwrite a source file')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'package': 'CT_Xray_GUI_Handover', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'purpose': 'research GUI inference handover', 'missing_optional': missing_optional, 'files': {}}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as z:
        for name, path in selected:
            z.write(path, arcname=name)
            manifest['files'][name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
            print('PACKED', name, flush=True)
        z.write(handover, arcname='HANDOVER.md')
        manifest['files']['HANDOVER.md'] = {'sha256': digest(handover), 'bytes': handover.stat().st_size}
        z.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(output) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f'ZIP CRC verification failed: {bad}')
    print(f'ZIP VERIFIED: {output} ({output.stat().st_size / (1024*1024):.1f} MiB)')
    print('Review contents before sharing; model files may be subject to data/license restrictions.')

if __name__ == '__main__':
    main()
