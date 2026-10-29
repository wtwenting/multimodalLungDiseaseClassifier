# BIMCV Dataset Download

This guide documents the download workflow found in the local BIMCV analysis scripts and provides a standalone archive downloader at `src/download_bimcv_dataset.py`.

The dataset contains medical CT and chest X-ray data. Confirm the BIMCV license, institutional approval, storage policy, and redistribution permissions before downloading or sharing it. Do not commit raw images, extracted patient directories, manifests containing patient IDs, or downloaded archives to Git.

## What the workflow does

The upstream scripts use two stages:

1. `plan_frontal_archives.py` reads local archive listings and selects missing patients with both frontal AP/PA X-ray metadata and CT metadata.
2. `download_from_plan.py` downloads only the archives needed by that plan, extracts selected members, verifies sizes, and optionally deletes verified archives.

The standalone helper in this repository is intentionally narrower: it downloads complete `.tar.gz` archive parts with resume support and gzip validation. It does not extract files or choose patients.

## Prerequisites

- macOS or Linux
- Python 3.9 or newer
- `curl` on `PATH`
- Enough free disk space for the selected archives plus a reserve
- An approved BIMCV/B2DROP account or access path, if the upstream service requires one

The downloader uses only the Python standard library. The analysis and preprocessing scripts require the packages in `requirements.txt`, but those packages are not needed just to download archives.

## Download all archive parts

From the `ensemble` directory:

```sh
python src/download_bimcv_dataset.py \
  --dir "$HOME/data/BIMCV-COVID-19-archives" \
  --min-free-gb 20 \
  --sha256
```

By default, the script downloads the known negative and positive archive parts listed in the upstream downloader. Existing verified files are skipped. Interrupted downloads remain as `.partial` files and resume on the next run. Verification markers are written beside successful archives.

Start with a small smoke test before a large download:

```sh
python src/download_bimcv_dataset.py \
  --dir "$HOME/data/BIMCV-COVID-19-archives" \
  --only negative \
  --max-archives 1 \
  --min-free-gb 20
```

## Use an existing quota plan

The planner from the upstream analysis repository creates a CSV such as:

```text
dataAnalysis/output/frontal_download/archives_to_download_filteredXrayView_CT.csv
```

The standalone helper reads the `COVID status` and `Archive to download` columns and downloads each archive once:

```sh
python src/download_bimcv_dataset.py \
  --plan /path/to/archives_to_download_filteredXrayView_CT.csv \
  --dir "$HOME/data/BIMCV-COVID-19-archives" \
  --min-free-gb 20 \
  --sha256
```

A plan controls archive selection only. It does not prove that every archive member is clinically suitable or that CT volumes are valid 3D scans.

## Existing upstream two-stage workflow

If you have the full `dataAnalysis` repository and its archive listings, use its patient-aware workflow instead:

```sh
python dataAnalysis/scripts/plan_frontal_archives.py \
  --additional-negative 100 \
  --additional-positive 50

python dataAnalysis/scripts/download_from_plan.py \
  --plan dataAnalysis/output/frontal_download/archives_to_download_filteredXrayView_CT.csv \
  --max-archives 1 \
  --min-free-gb 20 \
  --working-headroom-gb 16
```

Review the one-archive dry run first. Add `--apply` only when the destination, quota, and free-space policy are confirmed. The upstream extractor writes into its configured dataset root and can delete verified archives unless `--keep-archives` is supplied.

## Safety notes

- Keep archives and extracted data outside the Git repository.
- Do not use `git add .` in a directory containing BIMCV data.
- Keep the laptop connected to power for long transfers.
- Do not lower the free-space reserve without checking archive sizes and extraction requirements.
- Treat patient IDs and source paths as sensitive metadata.
- The downloader validates archive size and gzip integrity; it does not validate labels, CT geometry, lung anatomy, or clinical quality.
- Do not use this dataset or the resulting classifier for clinical decisions.
