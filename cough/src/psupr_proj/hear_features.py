"""Extract frozen Google HeAR embeddings for the existing COUGHVID split.

This module is intentionally isolated from the existing Model 1-3 pipeline.
It reads metadata_split.csv, loads each recording as 16 kHz mono, divides it
into non-overlapping 2-second clips, obtains frozen HeAR embeddings, and mean
pools clip embeddings into one 512-D recording embedding.

It supports a balanced smoke test and full extraction to compressed NPZ files.
No classifier training or resplitting is performed here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
EMBEDDING_DIM = 512
LABEL_ORDER = ("healthy", "COVID-19")
VALID_LABELS = set(LABEL_ORDER)
VALID_SPLITS = ("train", "validation", "test")


def load_records(metadata_path: Path, split: str) -> list[dict[str, str]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}")

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"uuid", "status", "audio_path", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

        return [
            row for row in reader
            if row["split"].strip() == split
            and row["status"].strip() in VALID_LABELS
        ]


def select_balanced(records: list[dict[str, str]], per_class: int) -> list[dict[str, str]]:
    """Deterministically select up to N rows per class, preserving file order."""
    selected: list[dict[str, str]] = []
    for label in LABEL_ORDER:
        class_rows = [r for r in records if r["status"].strip() == label]
        if not class_rows:
            raise ValueError(f"No records found for label {label!r}")
        selected.extend(class_rows[: min(per_class, len(class_rows))])
    return selected


def load_audio_16k(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Empty waveform: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite waveform values: {path}")
    return waveform


def make_hear_clips(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    n_clips = max(1, int(np.ceil(len(waveform) / CLIP_SAMPLES)))
    padded_length = n_clips * CLIP_SAMPLES
    padded = np.pad(waveform, (0, padded_length - len(waveform)))
    return padded.reshape(n_clips, CLIP_SAMPLES).astype(np.float32, copy=False)


def load_hear_model():
    try:
        from huggingface_hub import from_pretrained_keras
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "HeAR TensorFlow loading requires legacy tf_keras and "
            "huggingface_hub<1.0. Install them in an isolated environment, e.g. "
            "pip install 'huggingface_hub<1.0' tf_keras."
        ) from exc

    try:
        model = from_pretrained_keras("google/hear")
    except Exception as exc:
        raise RuntimeError(
            "Could not load google/hear. Log in to Hugging Face and accept the "
            "Health AI Developer Foundations terms on the google/hear model page."
        ) from exc

    return model.signatures["serving_default"]


def embed_recording(serving_signature, waveform: np.ndarray) -> tuple[np.ndarray, int]:
    clips = make_hear_clips(waveform)
    output = serving_signature(x=clips)["output_0"].numpy()
    output = np.asarray(output, dtype=np.float32)

    expected = (len(clips), EMBEDDING_DIM)
    if output.shape != expected:
        raise ValueError(f"Unexpected HeAR output shape {output.shape}; expected {expected}")
    if not np.isfinite(output).all():
        raise ValueError("HeAR produced non-finite embedding values")

    return output.mean(axis=0).astype(np.float32), len(clips)


def _extract_records(serving_signature, records: list[dict[str, str]], verbose: bool = True):
    embeddings: list[np.ndarray] = []
    uuids: list[str] = []
    labels: list[str] = []
    splits: list[str] = []
    clip_counts: list[int] = []
    durations: list[float] = []

    for index, record in enumerate(records, start=1):
        waveform = load_audio_16k(Path(record["audio_path"]))
        embedding, n_clips = embed_recording(serving_signature, waveform)
        embeddings.append(embedding)
        uuids.append(record["uuid"].strip())
        labels.append(record["status"].strip())
        splits.append(record["split"].strip())
        clip_counts.append(n_clips)
        durations.append(len(waveform) / SAMPLE_RATE)

        if verbose:
            print(
                f"[{index}/{len(records)}] uuid={uuids[-1]} | label={labels[-1]} | "
                f"duration={durations[-1]:.2f}s | clips={n_clips} | embedding={embedding.shape}"
            )

    batch = np.stack(embeddings).astype(np.float32)
    if batch.shape != (len(records), EMBEDDING_DIM):
        raise ValueError(f"Unexpected embedding batch shape: {batch.shape}")

    return {
        "embeddings": batch,
        "uuid": np.asarray(uuids),
        "label": np.asarray(labels),
        "split": np.asarray(splits),
        "clip_count": np.asarray(clip_counts, dtype=np.int32),
        "duration_seconds": np.asarray(durations, dtype=np.float32),
    }


def smoke_test(metadata_path: Path, split: str = "train", per_class: int = 5) -> None:
    records = load_records(metadata_path, split)
    if not records:
        raise ValueError(f"No eligible {split} records found in {metadata_path}")
    selected = select_balanced(records, per_class)

    print("\n" + "=" * 68)
    print("HeAR BALANCED FEATURE EXTRACTION SMOKE TEST")
    print("=" * 68)
    print(f"Metadata:        {metadata_path}")
    print(f"Split:           {split}")
    print(f"Per class:       up to {per_class}")
    print(f"Records tested:  {len(selected)}")
    print(f"HeAR input:      ({CLIP_SAMPLES},) = 2.0 s at 16 kHz")
    print(f"HeAR output:     {EMBEDDING_DIM}-D per clip")
    print("Aggregation:     mean across clips per recording")
    print("Training:        NONE")

    serving_signature = load_hear_model()
    result = _extract_records(serving_signature, selected)

    unique, counts = np.unique(result["label"], return_counts=True)
    print("\nPASS")
    print(f"Recording embedding batch: {result['embeddings'].shape}, dtype={result['embeddings'].dtype}")
    print("Class counts: " + ", ".join(f"{k}={v}" for k, v in zip(unique, counts)))
    print("Existing split/labels were reused; no training or resplitting occurred.")


def extract_all(metadata_path: Path, output_dir: Path) -> None:
    """Extract and save one NPZ per existing split using a single HeAR load."""
    output_dir.mkdir(parents=True, exist_ok=True)
    serving_signature = load_hear_model()

    print("\n" + "=" * 68)
    print("HeAR FULL FEATURE EXTRACTION")
    print("=" * 68)
    print(f"Metadata:    {metadata_path}")
    print(f"Output dir:  {output_dir}")
    print("Training:    NONE")
    print("Resplitting: NONE")

    for split in VALID_SPLITS:
        records = load_records(metadata_path, split)
        if not records:
            raise ValueError(f"No eligible {split} records found in {metadata_path}")

        print(f"\n--- {split.upper()} ({len(records)} recordings) ---")
        result = _extract_records(serving_signature, records)
        output_path = output_dir / f"hear_{split}.npz"
        np.savez_compressed(output_path, **result)

        # Immediate round-trip validation protects against incomplete/corrupt saves.
        with np.load(output_path) as check:
            if check["embeddings"].shape != (len(records), EMBEDDING_DIM):
                raise ValueError(f"Saved file validation failed: {output_path}")
            if len(check["uuid"]) != len(records) or len(check["label"]) != len(records):
                raise ValueError(f"Saved metadata validation failed: {output_path}")

        unique, counts = np.unique(result["label"], return_counts=True)
        print(f"Saved: {output_path}")
        print(f"Shape: {result['embeddings'].shape}")
        print("Class counts: " + ", ".join(f"{k}={v}" for k, v in zip(unique, counts)))

    print("\nPASS: all existing train/validation/test embeddings were extracted and verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/coughvid_20211012_processed/metadata_split.csv"),
        help="Path to the existing metadata_split.csv.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "extract"),
        default="smoke",
        help="smoke = balanced test; extract = save all three existing splits.",
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="train",
        help="Existing split used in smoke mode (default: train).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Smoke mode: number of recordings per class (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hear_embeddings"),
        help="Extract mode: directory for hear_train/validation/test.npz.",
    )
    args = parser.parse_args()

    if args.per_class < 1:
        parser.error("--per-class must be at least 1")

    if args.mode == "smoke":
        smoke_test(args.metadata, split=args.split, per_class=args.per_class)
    else:
        extract_all(args.metadata, args.output_dir)


if __name__ == "__main__":
    main()
