"""Spectral-energy EDA on validated Coswara heavy-cough recordings.

Run from project root:
 python coswara_spectral_energy_eda.py --metadata data/coswara/coswara_heavy_valid_metadata.csv --output-dir data/coswara/eda
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"count": 0}
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()), "min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "max": float(a.max())}


def measure(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("empty or non-finite waveform")
    x = x.mean(axis=1).astype(np.float64)
    if not np.any(x):
        raise ValueError("zero-energy waveform")
    # Parseval-correct one-sided power: double bins with negative-frequency partners.
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    if len(x) % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = power.sum()
    below = power[frequencies <= 8000].sum() / total * 100
    return int(sr), float(len(x) / sr), float(below), float(100 - below)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.metadata.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Metadata contains no rows')
    ids = [r['uuid'].strip() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate participant IDs in validated metadata')
    out = []
    errors = []
    for row in rows:
        path = Path(row['audio_path'])
        try:
            sr, duration, below, above = measure(path)
            out.append({'uuid': row['uuid'], 'status': row['status'],
                        'sample_rate_hz': sr, 'duration_seconds': duration,
                        'energy_at_or_below_8khz_pct': below,
                        'energy_above_8khz_pct': above,
                        'original_nyquist_hz': sr / 2,
                        'has_observable_above_8khz_band': sr > 16000})
        except (OSError, ValueError, RuntimeError) as e:
            errors.append({'uuid': row['uuid'], 'audio_path': str(path), 'error': str(e)})
    if out:
        with (args.output_dir / 'coswara_heavy_valid_spectral_energy.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0]))
            writer.writeheader()
            writer.writerows(out)
    by_class = defaultdict(list)
    sample_rates = Counter()
    for r in out:
        by_class[r['status']].append(r)
        sample_rates[r['sample_rate_hz']] += 1
    summary = {'metadata': str(args.metadata), 'requested_recordings': len(rows),
               'analyzed_recordings': len(out), 'failed_recordings': len(errors),
               'sample_rate_counts': dict(sorted(sample_rates.items())),
               'classes': {label: {'count': len(records),
                 'energy_at_or_below_8khz_pct': summarize([r['energy_at_or_below_8khz_pct'] for r in records]),
                 'energy_above_8khz_pct': summarize([r['energy_above_8khz_pct'] for r in records]),
                 'recordings_with_original_sr_above_16khz': sum(r['has_observable_above_8khz_band'] for r in records)}
                 for label, records in sorted(by_class.items())},
               'failures': errors,
               'interpretation_note': 'High energy retention is not proof that frequencies above 8 kHz lack discriminative information. Recordings originally sampled at 16 kHz or lower cannot reveal energy above 8 kHz.'}
    (args.output_dir / 'coswara_heavy_valid_spectral_energy_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'failures'}, indent=2))
    if errors:
        print(f'WARNING: {len(errors)} recordings could not be analyzed; see JSON failures')


if __name__ == '__main__':
    main()
