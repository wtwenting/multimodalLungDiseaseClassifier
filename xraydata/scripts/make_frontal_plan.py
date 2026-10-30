#!/usr/bin/env python3
"""Pick X-ray-only patients that have a frontal (AP/PA) X-ray and plan which BIMCV archives to download.

Reads (inside --work):
  all_patients_modalities.csv   columns: Patient ID, COVID status, Decision, ...  (Decision == KEEP = X-ray-only)
  listings/*tvf.txt             one 'tar tvf' listing per archive
Writes:
  frontal_xray_only_plan.csv    columns: COVID status, Archive to download, Selected patient IDs, Frontal images

Example:
  python make_frontal_plan.py --work C:\\BIMCV_xray\\xray_work --positive 1000 --negative 1000
"""
import argparse, csv, re, collections, pathlib

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--work', type=pathlib.Path, default=pathlib.Path('xray_work'), help='folder with listings/ and all_patients_modalities.csv')
ap.add_argument('--positive', type=int, default=1000, help='target COVID-positive patients')
ap.add_argument('--negative', type=int, default=1000, help='target COVID-negative patients')
args = ap.parse_args()

work = args.work
N_PATIENTS = {'Positive': args.positive, 'Negative': args.negative}
status = {r['Patient ID']: r['COVID status']
          for r in csv.DictReader(open(work/'all_patients_modalities.csv', encoding='utf-8-sig'))
          if r['Decision'] == 'KEEP'}                       # X-ray-only patients

# archive -> {patient: [frontal PNG members]}
frontal = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(work.glob('listings/*tvf.txt')):
    m = re.search(r'covid19_(?:posi|pos|neg)_subjects_part[a-z]+', f.name.lower())
    if not m:
        continue
    archive = m.group(0) + '.tar.gz'
    for line in f.read_text(errors='replace').split('\n'):
        member = line.split()[-1] if line.strip() else ''
        name = member.lower()
        if not re.search(r'_vp-(ap|pa)_.*(cr|dx)\.png$', name): continue
        pid = re.search(r'sub-(s\d+)', name).group(1).upper()
        if pid in status:
            frontal[archive][pid].append(member)

plan, seen = [], set()
for cls in ('Positive', 'Negative'):
    archs = sorted((a for a in frontal if any(status[p] == cls for p in frontal[a])),
                   key=lambda a: -len(frontal[a]))           # most patients first
    chosen = 0
    for a in archs:
        if chosen >= N_PATIENTS[cls]: break
        pids = sorted(p for p in frontal[a] if status[p] == cls and p not in seen)
        if not pids: continue
        seen.update(pids); chosen += len(pids)
        plan.append({'COVID status': cls, 'Archive to download': a,
                     'Selected patient IDs': ';'.join(pids),
                     'Frontal images': sum(len(frontal[a][p]) for p in pids)})
    n_img = sum(r['Frontal images'] for r in plan if r['COVID status'] == cls)
    n_arc = sum(r['COVID status'] == cls for r in plan)
    print(f'{cls}: {chosen} patients, {n_img} AP/PA images, {n_arc} archives')

out = work/'frontal_xray_only_plan.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(plan[0])); w.writeheader(); w.writerows(plan)
print('Plan saved:', out)
