#!/usr/bin/env python3
"""Prepare Common Voice Malaysian English for Qwen3-ASR fine-tuning.

Reads existing train/dev/test TSVs, resamples clips to 16 kHz mono WAV,
and writes JSONL with the official language-tag format:

  {"audio": "/abs/path.wav", "text": "language English<asr_text>..."}

Example:
  python prep_data.py \\
    --src "/Users/sinh/Downloads/1784486741842-cv26-malaysian-english" \\
    --out ./data_ms_en
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

LANGUAGE_TAG = "language English<asr_text>"
SAMPLING_RATE = 16000
MIN_AUDIO_SECONDS = 0.3
MAX_AUDIO_SECONDS = 30.0


def load_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def process_split(
    rows: list[dict],
    clips_dir: Path,
    wav_dir: Path,
    out_jsonl: Path,
    split_name: str,
) -> dict:
    wav_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "saved": 0,
        "missing": 0,
        "empty_text": 0,
        "too_short": 0,
        "too_long": 0,
        "bad_audio": 0,
        "hours": 0.0,
    }

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(tqdm(rows, desc=split_name)):
            sentence = (row.get("sentence") or "").strip()
            if not sentence:
                stats["empty_text"] += 1
                continue

            clip_name = row["path"]
            src = clips_dir / clip_name
            if not src.exists():
                stats["missing"] += 1
                continue

            try:
                wav, _ = librosa.load(str(src), sr=SAMPLING_RATE, mono=True)
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            except Exception:
                stats["bad_audio"] += 1
                continue

            duration = len(wav) / SAMPLING_RATE
            if duration < MIN_AUDIO_SECONDS:
                stats["too_short"] += 1
                continue
            if duration > MAX_AUDIO_SECONDS:
                stats["too_long"] += 1
                continue
            if (
                np.isnan(wav).any()
                or np.isinf(wav).any()
                or float(np.abs(wav).max()) < 1e-8
            ):
                stats["bad_audio"] += 1
                continue

            stem = Path(clip_name).stem
            wav_name = f"{split_name}_{stem}.wav"
            wav_path = wav_dir / wav_name
            sf.write(str(wav_path), wav, SAMPLING_RATE)

            # Relative to --out so the same JSONL works locally and on Modal volumes.
            record = {
                "audio": f"wavs/{wav_name}",
                "text": f"{LANGUAGE_TAG}{sentence}",
                "duration": round(duration, 3),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["saved"] += 1
            stats["hours"] += duration / 3600.0

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Prep Malaysian English CV for Qwen3-ASR")
    p.add_argument(
        "--src",
        type=str,
        required=True,
        help="Common Voice folder with train/dev/test.tsv and clips/",
    )
    p.add_argument("--out", type=str, default="./data_ms_en")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    clips_dir = src / "clips"
    wav_dir = out / "wavs"
    out.mkdir(parents=True, exist_ok=True)

    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing clips dir: {clips_dir}")

    print(f"Source: {src}")
    print(f"Output: {out}")

    for split in ("train", "dev", "test"):
        tsv = src / f"{split}.tsv"
        if not tsv.exists():
            raise FileNotFoundError(f"Missing {tsv}")
        rows = load_tsv(tsv)
        jsonl = out / f"{split}.jsonl"
        stats = process_split(rows, clips_dir, wav_dir, jsonl, split)
        print(
            f"[{split}] saved={stats['saved']} hours={stats['hours']:.3f} "
            f"missing={stats['missing']} empty={stats['empty_text']} "
            f"short={stats['too_short']} long={stats['too_long']} bad={stats['bad_audio']}"
        )
        print(f"  -> {jsonl}")

    # Marker used by Modal upload / path rewriting
    meta = {
        "language_tag": LANGUAGE_TAG,
        "sampling_rate": SAMPLING_RATE,
        "source": str(src),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    # Avoid tokenizer fork warnings if imported alongside HF later
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
