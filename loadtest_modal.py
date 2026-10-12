#!/usr/bin/env python3
"""Baseline concurrent-stream load test for original Qwen3-ASR-1.7B on Modal.

Simulates multiple simultaneous ASR requests by running waves of
``model.transcribe(audio=[…])`` at increasing concurrency (batch size):

    1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → … (doubles until OOM / max)

For each concurrency level records:
  Concurrent Streams | Avg RTF | P50 RTF | P95 RTF |
  GPU Utilization | VRAM Usage | Throughput | WER | WER (no punct)

Throughput / RTF / GPU use ``--waves`` (load averaging).
WER uses a fixed ``--wer-set-size`` clip list (identical at every concurrency),
batched at that level's stream count — so WER is comparable across rows.
Pass ``--skip-wer`` to measure throughput/GPU only (much faster).

WER matches training eval: both with punctuation kept and with punctuation stripped.
This is the **unoptimized baseline** (single model, native qwen-asr batching).
A later optimized serving path can reuse the same ladder for comparison.

Prerequisites
-------------
1) Prepared data uploaded to the Modal volume::

    export MODAL_PROFILE=...
    modal run train_modal.py --action upload --local-dir ./data_ms_en

2) Run the load test::

    modal run loadtest_modal.py
    modal run loadtest_modal.py --max-concurrency 256 --waves 4 --wer-set-size 128
    modal run loadtest_modal.py --skip-wer   # throughput / GPU only

Outputs
-------
  /vol/checkpoints/baseline_loadtest/loadtest_baseline.json
  /vol/checkpoints/baseline_loadtest/loadtest_baseline_summary.json
  ./loadtest_baseline_summary.json  (local mirror)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import modal

load_dotenv(Path(__file__).resolve().parent / ".env")
MODAL_PROFILE = os.environ.get("MODAL_PROFILE", "").strip()

APP_NAME = "qwen3-asr-ms-en-loadtest"
VOLUME_NAME = "qwen3-asr-ms-en"
VOL_ROOT = "/vol"
DATA_ROOT = f"{VOL_ROOT}/data"
CKPT_ROOT = f"{VOL_ROOT}/checkpoints"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.51.0",
        "accelerate",
        "peft>=0.14.0",
        "datasets",
        "soundfile",
        "librosa",
        "jiwer",
        "numpy",
        "tqdm",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "qwen-asr",
        "python-dotenv",
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


# ── Helpers (run inside Modal containers) ────────────────────────────────────


def _log(msg: str) -> None:
    """Print immediately so Modal streams stage progress to the local terminal."""
    print(msg, flush=True)


def _resolve_audio_path(path: str, data_root: str) -> str:
    import os

    if os.path.isabs(path) and os.path.exists(path):
        return path
    return os.path.join(data_root, path)


def _load_jsonl(path: str) -> list[dict]:
    import json

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str, obj) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    _log(f"[write] {path}")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def _collect_env(precision: str, framework: str) -> dict:
    import os
    import platform
    import subprocess

    import torch

    gpu_name = None
    gpu_mem_total_gb = None
    cuda_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem_total_gb = round(props.total_memory / (1024**3), 2)
        cuda_version = torch.version.cuda

    driver_version = None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if out:
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 1 and not gpu_name:
                gpu_name = parts[0]
            if len(parts) >= 2:
                try:
                    gpu_mem_total_gb = round(float(parts[1]) / 1024.0, 2)
                except ValueError:
                    pass
            if len(parts) >= 3:
                driver_version = parts[2]
    except Exception:
        pass

    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    ram_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    ram_gb = round(kb / (1024**2), 2)
                    break
    except OSError:
        pass

    return {
        "gpu_model": gpu_name,
        "gpu_memory_gb": gpu_mem_total_gb,
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "cuda_version": cuda_version,
        "nvidia_driver": driver_version,
        "pytorch_version": torch.__version__,
        "inference_framework": framework,
        "precision": precision,
        "attn_implementation": "sdpa",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "modal_gpu_request": GPU,
    }


def _audio_duration_sec(path: str) -> float:
    import soundfile as sf

    return float(sf.info(path).duration)


def _normalize_for_wer(text: str, *, strip_punct: bool = False) -> str:
    """Strip Qwen language prefix; lowercase; optionally drop punctuation."""
    import re
    import string

    text = text or ""
    m = re.search(r"<asr_text>(.*)$", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    text = text.lower()
    if strip_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _compute_wer(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus WER with punctuation kept (`wer`) and stripped (`wer_no_punct`)."""
    from jiwer import process_words

    kept_refs, kept_hyps = [], []
    kept_refs_np, kept_hyps_np = [], []
    for ref_raw, hyp_raw in zip(references, hypotheses):
        ref_n = _normalize_for_wer(ref_raw, strip_punct=False)
        hyp_n = _normalize_for_wer(hyp_raw, strip_punct=False)
        ref_np = _normalize_for_wer(ref_raw, strip_punct=True)
        hyp_np = _normalize_for_wer(hyp_raw, strip_punct=True)
        if ref_n.strip():
            kept_refs.append(ref_n)
            kept_hyps.append(hyp_n)
            kept_refs_np.append(ref_np)
            kept_hyps_np.append(hyp_np)

    def _corpus(refs: list[str], hyps: list[str]) -> dict:
        if not refs:
            return {
                "wer": 1.0,
                "n": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
            }
        out = process_words(refs, hyps)
        return {
            "wer": float(out.wer),
            "n": len(refs),
            "substitutions": int(out.substitutions),
            "deletions": int(out.deletions),
            "insertions": int(out.insertions),
            "hits": int(out.hits),
        }

    with_p = _corpus(kept_refs, kept_hyps)
    no_p = _corpus(kept_refs_np, kept_hyps_np)
    return {
        "wer": with_p["wer"],
        "wer_no_punct": no_p["wer"],
        "n": with_p["n"],
        "substitutions": with_p["substitutions"],
        "deletions": with_p["deletions"],
        "insertions": with_p["insertions"],
        "hits": with_p["hits"],
        "substitutions_no_punct": no_p["substitutions"],
        "deletions_no_punct": no_p["deletions"],
        "insertions_no_punct": no_p["insertions"],
        "hits_no_punct": no_p["hits"],
    }


def _select_loadtest_samples(
    jsonl_paths: list[str],
    data_root: str,
    pool_size: int = 128,
    seed: int = 42,
    prefer_bucket: str = "5-15s",
) -> list[dict]:
    """Build a reusable audio pool (prefer mid-length clips for stable streams)."""
    import os
    import random

    preferred: list[dict] = []
    fallback: list[dict] = []

    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            _log(f"[loadtest] skip missing jsonl: {jsonl_path}")
            continue
        rows = _load_jsonl(jsonl_path)
        _log(f"[loadtest] scanning {os.path.basename(jsonl_path)} ({len(rows)} rows)…")
        kept = 0
        for row in rows:
            path = _resolve_audio_path(row["audio"], data_root)
            if not os.path.exists(path):
                continue
            # Prefer duration from JSONL (prep_data writes it) — avoid opening every WAV.
            if "duration" in row:
                dur = float(row["duration"])
            else:
                dur = _audio_duration_sec(path)
            if dur < 2.0 or dur > 30.0:
                continue
            item = {
                "audio": row["audio"],
                "path": path,
                "duration_sec": dur,
                "text": row.get("text") or "",
                "source_jsonl": os.path.basename(jsonl_path),
            }
            if prefer_bucket == "5-15s" and 5.0 <= dur < 15.0:
                preferred.append(item)
            elif prefer_bucket == "2-5s" and 2.0 <= dur < 5.0:
                preferred.append(item)
            else:
                fallback.append(item)
            kept += 1
        _log(f"[loadtest]   kept={kept} from {os.path.basename(jsonl_path)}")

    rng = random.Random(seed)
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    pool = (preferred + fallback)[:pool_size]
    if not pool:
        raise RuntimeError("No load-test audio samples found in prepared data")
    _log(
        f"[loadtest] audio pool: n={len(pool)} "
        f"(preferred={len(preferred)}, fallback={len(fallback)}, "
        f"prefer={prefer_bucket})"
    )
    return pool


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "oom",
    )
    return any(n in msg for n in needles) or "outofmemory" in name


def _gpu_poller(stop_event, util_samples: list[float], mem_used_mib: list[float]):
    import subprocess
    import threading

    def _poll():
        while not stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                line = out.splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1:
                    util_samples.append(float(parts[0]))
                if len(parts) >= 2:
                    mem_used_mib.append(float(parts[1]))
            except Exception:
                pass
            stop_event.wait(0.25)

    t = threading.Thread(target=_poll, daemon=True)
    return t


def _build_concurrency_ladder(max_concurrency: int) -> list[int]:
    """1,2,4,… up through max_concurrency (include max even if not a power of 2)."""
    levels: list[int] = []
    c = 1
    while c <= max_concurrency:
        levels.append(c)
        c *= 2
    if levels[-1] != max_concurrency:
        levels.append(max_concurrency)
    return levels


def _transcribe_batch(model, paths: list[str]):
    """Transcribe one batch; re-raise OOM, return (results, proc_sec)."""
    import time

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model.transcribe(audio=paths)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return results, time.perf_counter() - t0


def _run_wer_on_fixed_set(
    model,
    wer_set: list[dict],
    concurrency: int,
) -> tuple[dict, list[dict], list[str]]:
    """Score the same fixed clips at this concurrency (batches of size C, last may be smaller)."""
    refs: list[str] = []
    hyps: list[str] = []
    per_utt: list[dict] = []
    errors: list[str] = []

    if not wer_set:
        return _compute_wer([], []), [], ["empty wer_set"]

    n = len(wer_set)
    batch_size = max(1, min(concurrency, n))
    n_batches = (n + batch_size - 1) // batch_size
    _log(
        f"[loadtest]   wer pass: n={n} batch_size={batch_size} "
        f"batches={n_batches} (fixed set)"
    )

    for bi, start in enumerate(range(0, n, batch_size)):
        batch = wer_set[start : start + batch_size]
        paths = [s["path"] for s in batch]
        try:
            results, proc = _transcribe_batch(model, paths)
        except Exception as exc:
            if _is_oom(exc):
                raise
            errors.append(f"wer_batch={bi}: {type(exc).__name__}: {exc}")
            _log(f"[loadtest]   wer batch {bi + 1}/{n_batches} error: {exc}")
            continue
        _log(f"[loadtest]   wer batch {bi + 1}/{n_batches} done in {proc:.2f}s")

        audio_dur = float(sum(s["duration_sec"] for s in batch))
        for s, res in zip(batch, results):
            hyp = getattr(res, "text", "") or ""
            refs.append(s.get("text") or "")
            hyps.append(hyp)
            per_utt.append(
                {
                    "audio": s["audio"],
                    "duration_sec": s["duration_sec"],
                    "ref": s.get("text") or "",
                    "hyp": hyp,
                    "hyp_chars": len(hyp),
                }
            )

    return _compute_wer(refs, hyps), per_utt, errors


def _run_concurrency_level(
    model,
    pool: list[dict],
    wer_set: list[dict],
    concurrency: int,
    waves: int,
    skip_wer: bool = False,
) -> dict:
    """Throughput via ``waves``; optional WER via fixed ``wer_set``."""
    import threading
    import time

    import torch

    thr_pool = pool
    if concurrency > len(thr_pool):
        # Cycle so large concurrency still has enough paths for throughput waves
        reps = (concurrency // len(thr_pool)) + 1
        thr_pool = (thr_pool * reps)[: max(concurrency * waves, concurrency)]

    util_samples: list[float] = []
    mem_used_mib: list[float] = []
    stop_poll = threading.Event()
    poller = _gpu_poller(stop_poll, util_samples, mem_used_mib)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        poller.start()

    rtfs: list[float] = []
    durs: list[float] = []
    per_utt: list[dict] = []
    wave_times: list[float] = []
    total_audio = 0.0
    errors: list[str] = []

    # ── Throughput / RTF / GPU (waves) ─────────────────────────────────────
    wall_t0 = time.perf_counter()
    try:
        for wave_i in range(waves):
            batch = []
            for j in range(concurrency):
                batch.append(thr_pool[(wave_i * concurrency + j) % len(thr_pool)])
            paths = [s["path"] for s in batch]
            audio_dur = float(sum(s["duration_sec"] for s in batch))
            total_audio += audio_dur

            _log(
                f"[loadtest]   wave {wave_i + 1}/{waves} "
                f"streams={concurrency} audio={audio_dur:.1f}s …"
            )
            try:
                results, proc = _transcribe_batch(model, paths)
            except Exception as exc:
                if _is_oom(exc):
                    raise
                errors.append(f"wave={wave_i}: {type(exc).__name__}: {exc}")
                _log(f"[loadtest]   wave {wave_i + 1} error: {exc}")
                continue
            wave_times.append(proc)
            _log(f"[loadtest]   wave {wave_i + 1}/{waves} done in {proc:.2f}s")

            for s, res in zip(batch, results):
                hyp = getattr(res, "text", "") or ""
                share = (
                    (s["duration_sec"] / audio_dur) * proc
                    if audio_dur > 0
                    else proc / len(batch)
                )
                rtf = share / s["duration_sec"] if s["duration_sec"] > 0 else float("nan")
                rtfs.append(rtf)
                durs.append(s["duration_sec"])
                per_utt.append(
                    {
                        "wave": wave_i,
                        "audio": s["audio"],
                        "duration_sec": s["duration_sec"],
                        "proc_sec": share,
                        "rtf": rtf,
                        "hyp_chars": len(hyp),
                    }
                )
    finally:
        stop_poll.set()
        if poller.is_alive():
            poller.join(timeout=2.0)

    wall = time.perf_counter() - wall_t0
    peak_alloc_gb = None
    peak_reserved_gb = None
    if torch.cuda.is_available():
        peak_alloc_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        peak_reserved_gb = round(torch.cuda.max_memory_reserved() / (1024**3), 3)

    throughput_audio = (total_audio / wall) if wall > 0 else float("nan")
    throughput_utt = (len(rtfs) / wall) if wall > 0 else float("nan")

    # ── Fair WER on fixed set (identical clips every concurrency) ──────────
    if skip_wer:
        _log("[loadtest]   wer pass: skipped (--skip-wer)")
        wer_stats = None
        wer_per_utt: list[dict] = []
    else:
        wer_stats, wer_per_utt, wer_errors = _run_wer_on_fixed_set(
            model, wer_set, concurrency
        )
        errors.extend(wer_errors)

    def _wer_field(key: str, scale: float = 1.0):
        if wer_stats is None:
            return None
        return wer_stats[key] * scale

    return {
        "concurrent_streams": concurrency,
        "waves": waves,
        "skip_wer": skip_wer,
        "wer_set_size": 0 if skip_wer else len(wer_set),
        "n_utt": len(rtfs),
        "avg_rtf": float(sum(rtfs) / len(rtfs)) if rtfs else float("nan"),
        "p50_rtf": _percentile(rtfs, 50),
        "p95_rtf": _percentile(rtfs, 95),
        "min_rtf": float(min(rtfs)) if rtfs else float("nan"),
        "max_rtf": float(max(rtfs)) if rtfs else float("nan"),
        "total_audio_sec": total_audio,
        "wall_sec": wall,
        "mean_wave_sec": float(sum(wave_times) / len(wave_times)) if wave_times else float("nan"),
        "throughput_audio_sec_per_sec": throughput_audio,
        "throughput_utt_per_sec": throughput_utt,
        "wer": _wer_field("wer"),
        "wer_pct": _wer_field("wer", 100.0),
        "wer_no_punct": _wer_field("wer_no_punct"),
        "wer_no_punct_pct": _wer_field("wer_no_punct", 100.0),
        "wer_n": None if wer_stats is None else wer_stats["n"],
        "wer_substitutions": _wer_field("substitutions"),
        "wer_deletions": _wer_field("deletions"),
        "wer_insertions": _wer_field("insertions"),
        "wer_hits": _wer_field("hits"),
        "wer_substitutions_no_punct": _wer_field("substitutions_no_punct"),
        "wer_deletions_no_punct": _wer_field("deletions_no_punct"),
        "wer_insertions_no_punct": _wer_field("insertions_no_punct"),
        "wer_hits_no_punct": _wer_field("hits_no_punct"),
        "gpu_util_avg_pct": (
            float(sum(util_samples) / len(util_samples)) if util_samples else None
        ),
        "gpu_util_p50_pct": _percentile(util_samples, 50) if util_samples else None,
        "gpu_util_p95_pct": _percentile(util_samples, 95) if util_samples else None,
        "gpu_util_max_pct": float(max(util_samples)) if util_samples else None,
        "vram_used_avg_mib": (
            float(sum(mem_used_mib) / len(mem_used_mib)) if mem_used_mib else None
        ),
        "vram_used_max_mib": float(max(mem_used_mib)) if mem_used_mib else None,
        "torch_peak_allocated_gb": peak_alloc_gb,
        "torch_peak_reserved_gb": peak_reserved_gb,
        "errors": errors,
        "ok": len(rtfs) > 0 and not errors,
        "per_utt": per_utt,
        "wer_per_utt": wer_per_utt,
    }


def _format_table(levels: list[dict]) -> str:
    """ASCII table matching the requested load-test report columns."""
    headers = (
        "Concurrent Streams",
        "Avg RTF",
        "P50 RTF",
        "P95 RTF",
        "GPU avg %",
        "GPU peak %",
        "VRAM avg (MiB)",
        "VRAM peak (MiB)",
        "Throughput (audio-s/s)",
        "WER %",
        "WER no-punct %",
    )
    rows: list[list[str]] = []
    for lv in levels:
        if lv.get("status") in ("oom", "error"):
            tag = "OOM" if lv.get("status") == "oom" else "ERR"
            rows.append(
                [str(lv["concurrent_streams"]), tag, tag, tag]
                + ["—"] * (len(headers) - 4)
            )
            continue

        def _f(x, nd=3):
            if x is None:
                return "—"
            try:
                if x != x:  # NaN
                    return "—"
                return f"{float(x):.{nd}f}"
            except (TypeError, ValueError):
                return "—"

        rows.append(
            [
                str(lv["concurrent_streams"]),
                _f(lv.get("avg_rtf"), 4),
                _f(lv.get("p50_rtf"), 4),
                _f(lv.get("p95_rtf"), 4),
                _f(lv.get("gpu_util_avg_pct"), 1),
                _f(lv.get("gpu_util_max_pct"), 1),
                _f(lv.get("vram_used_avg_mib"), 0),
                _f(lv.get("vram_used_max_mib"), 0),
                _f(lv.get("throughput_audio_sec_per_sec"), 2),
                _f(lv.get("wer_pct"), 2),
                _f(lv.get("wer_no_punct_pct"), 2),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(list(headers)), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _loadtest_inference(
    model_path: str,
    pool: list[dict],
    dtype,
    device: str = "cuda:0",
    warmup: int = 3,
    waves: int = 4,
    max_concurrency: int = 256,
    wer_set_size: int = 128,
    skip_wer: bool = False,
) -> dict:
    """Ladder concurrent streams; stop on OOM (or after max)."""
    import time

    import torch
    from qwen_asr import Qwen3ASRModel

    if skip_wer:
        wer_set: list[dict] = []
        _log("[loadtest] WER disabled (--skip-wer)")
    else:
        if wer_set_size < 1:
            raise ValueError(f"wer_set_size must be >= 1, got {wer_set_size}")
        if wer_set_size > len(pool):
            raise ValueError(
                f"wer_set_size={wer_set_size} > pool_size={len(pool)}; "
                f"increase --pool-size or lower --wer-set-size"
            )
        wer_set = pool[:wer_set_size]
        _log(
            f"[loadtest] fixed WER set: n={len(wer_set)} "
            f"(identical clips at every concurrency)"
        )

    precision = "bfloat16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
    _log("[loadtest] stage: collect environment")
    env = _collect_env(
        precision=precision,
        framework="qwen-asr (Qwen3ASRModel.transcribe, concurrent batch)",
    )
    _log("[loadtest] environment:")
    for k, v in env.items():
        _log(f"  {k}: {v}")

    _log(f"[loadtest] stage: load model {model_path} (HF download may take a few min)…")
    load_t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_sec = time.perf_counter() - load_t0
    _log(f"[loadtest] model loaded in {load_sec:.2f}s")

    warm_n = min(warmup, len(pool))
    _log(f"[loadtest] stage: warmup n={warm_n}")
    for i, s in enumerate(pool[:warm_n], start=1):
        _log(f"[loadtest]   warmup {i}/{warm_n}")
        _ = model.transcribe(audio=[s["path"]])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _log("[loadtest] warmup done")

    # 1→2→4→…→64 and keep doubling until max_concurrency or OOM
    ladder = _build_concurrency_ladder(max_concurrency)
    _log(f"[loadtest] stage: concurrency ladder {ladder}")
    _log(f"[loadtest] waves per level (throughput): {waves}")
    if skip_wer:
        _log("[loadtest] wer: skipped")
    else:
        _log(f"[loadtest] wer_set_size (fair WER): {len(wer_set)}")

    levels: list[dict] = []
    stopped_reason = None

    for concurrency in ladder:
        _log(f"\n[loadtest] === concurrent_streams={concurrency} ===")
        try:
            result = _run_concurrency_level(
                model=model,
                pool=pool,
                wer_set=wer_set,
                concurrency=concurrency,
                waves=waves,
                skip_wer=skip_wer,
            )
        except Exception as exc:
            status = "oom" if _is_oom(exc) else "error"
            _log(f"[loadtest] level {concurrency} failed ({status}): {exc}")
            levels.append(
                {
                    "concurrent_streams": concurrency,
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                    "ok": False,
                }
            )
            stopped_reason = status
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Stop ladder on OOM; hardware limit reached
            if status == "oom":
                break
            # Non-OOM: still stop — further levels likely fail the same way
            break

        result["status"] = "ok"
        # Drop bulky per-utt from the level summary kept in `levels` print path;
        # full detail stays in the returned report.
        levels.append(result)
        wer_msg = "wer=skipped"
        if result.get("wer_pct") is not None:
            wer_msg = (
                f"wer={result['wer_pct']:.2f}% "
                f"wer_no_punct={result['wer_no_punct_pct']:.2f}% "
                f"(wer_n={result['wer_n']}, fixed set)"
            )
        _log(
            f"[loadtest] streams={concurrency} "
            f"avg_rtf={result['avg_rtf']:.4f} "
            f"p50={result['p50_rtf']:.4f} "
            f"p95={result['p95_rtf']:.4f} "
            f"gpu_avg={result['gpu_util_avg_pct']} "
            f"gpu_peak={result['gpu_util_max_pct']} "
            f"vram_avg_mib={result['vram_used_avg_mib']} "
            f"vram_peak_mib={result['vram_used_max_mib']} "
            f"throughput={result['throughput_audio_sec_per_sec']:.2f} audio-s/s "
            f"{wer_msg}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = _format_table(levels)
    _log("\n=== Load test results (baseline) ===")
    _log(table)
    if stopped_reason:
        _log(f"\n[loadtest] stopped early: {stopped_reason}")

    # Compact rows for summary (no bulky per-utt lists)
    level_summaries = []
    for lv in levels:
        compact = {
            k: v for k, v in lv.items() if k not in ("per_utt", "wer_per_utt")
        }
        level_summaries.append(compact)

    return {
        "model": model_path,
        "variant": "baseline",
        "warmup": warmup,
        "waves": waves,
        "skip_wer": skip_wer,
        "wer_set_size": 0 if skip_wer else len(wer_set),
        "max_concurrency": max_concurrency,
        "concurrency_ladder": ladder,
        "model_load_sec": load_sec,
        "environment": env,
        "pool_size": len(pool),
        "levels": level_summaries,
        "levels_detail": levels,
        "table": table,
        "stopped_reason": stopped_reason,
    }


# ── Modal remote function ────────────────────────────────────────────────────


@app.function(
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60 * 4,
    memory=32768,
)
def run_loadtest(
    waves: int = 4,
    warmup: int = 3,
    seed: int = 42,
    pool_size: int = 128,
    wer_set_size: int = 128,
    skip_wer: bool = False,
    max_concurrency: int = 256,
    output_name: str = "baseline_loadtest",
) -> dict:
    """Baseline concurrent-stream load test for original Qwen3-ASR-1.7B."""
    import os

    import torch

    _log("[loadtest] stage: start remote worker")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", f"{CKPT_ROOT}/hf_cache")
    os.makedirs(CKPT_ROOT, exist_ok=True)

    train_src = f"{DATA_ROOT}/train.jsonl"
    dev_src = f"{DATA_ROOT}/dev.jsonl"
    test_src = f"{DATA_ROOT}/test.jsonl"
    _log("[loadtest] stage: check data on volume")
    for p in (train_src, test_src):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Upload prepared data first:\n"
                f"  export MODAL_PROFILE=<your-profile>\n"
                f"  modal run train_modal.py --action upload --local-dir ./data_ms_en"
            )
        _log(f"[loadtest]   found {p}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    _log(f"[loadtest] dtype={dtype} cuda={torch.cuda.is_available()}")

    # Throughput pool must be large enough to carve out the fixed WER set.
    effective_pool = pool_size if skip_wer else max(pool_size, wer_set_size)
    if effective_pool != pool_size:
        _log(
            f"[loadtest] bumping pool_size {pool_size} → {effective_pool} "
            f"to cover wer_set_size={wer_set_size}"
        )

    _log("[loadtest] stage: select audio pool")
    pool = _select_loadtest_samples(
        jsonl_paths=[test_src, dev_src, train_src],
        data_root=DATA_ROOT,
        pool_size=effective_pool,
        seed=seed,
        prefer_bucket="5-15s",
    )

    report = _loadtest_inference(
        model_path=MODEL_ID,
        pool=pool,
        dtype=dtype,
        device="cuda:0",
        warmup=warmup,
        waves=waves,
        max_concurrency=max_concurrency,
        wer_set_size=wer_set_size,
        skip_wer=skip_wer,
    )

    _log("[loadtest] stage: write results")
    out_dir = f"{CKPT_ROOT}/{output_name}"
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "loadtest_baseline.json"), report)
    summary = {k: v for k, v in report.items() if k != "levels_detail"}
    _write_json(os.path.join(out_dir, "loadtest_baseline_summary.json"), summary)
    vol.commit()
    _log(f"[done] loadtest → {os.path.join(out_dir, 'loadtest_baseline.json')}")
    return summary


# ── Local entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    waves: int = 4,
    warmup: int = 3,
    seed: int = 42,
    pool_size: int = 128,
    wer_set_size: int = 128,
    skip_wer: bool = False,
    max_concurrency: int = 256,
    output_name: str = "baseline_loadtest",
):
    """Run baseline concurrent ASR load test on Modal (L4)."""
    import json
    import os
    import subprocess
    import sys

    expected = MODAL_PROFILE or os.environ.get("MODAL_PROFILE", "").strip()
    cur = subprocess.run(
        [sys.executable, "-m", "modal", "profile", "current"],
        capture_output=True,
        text=True,
    )
    active = (cur.stdout or "").strip()
    if expected and active and active != expected:
        print(
            f"WARNING: active Modal profile is '{active}', "
            f"expected '{expected}' (MODAL_PROFILE).\n"
            f"  Run: export MODAL_PROFILE={expected}\n"
            f"   or: modal profile activate {expected}"
        )
    elif not expected:
        print(
            "WARNING: MODAL_PROFILE is unset. "
            "Set it in .env or the environment before modal commands."
        )

    result = run_loadtest.remote(
        waves=waves,
        warmup=warmup,
        seed=seed,
        pool_size=pool_size,
        wer_set_size=wer_set_size,
        skip_wer=skip_wer,
        max_concurrency=max_concurrency,
        output_name=output_name,
    )

    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "loadtest_baseline_summary.json",
    )
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[local] wrote {local_path}")

    print("\n=== Environment ===")
    env = result.get("environment") or {}
    for k in (
        "gpu_model",
        "gpu_memory_gb",
        "cuda_version",
        "pytorch_version",
        "inference_framework",
        "precision",
    ):
        print(f"  {k}: {env.get(k)}")

    print("\n=== Concurrent stream ladder (baseline) ===")
    print(result.get("table") or "(no table)")
    if result.get("stopped_reason"):
        print(f"\nStopped early: {result['stopped_reason']}")

    print("\nPer-level detail:")
    for lv in result.get("levels") or []:
        status = lv.get("status", "ok")
        if status != "ok":
            print(f"  streams={lv.get('concurrent_streams')}: {status} — {lv.get('error')}")
            continue
        wer_pct = lv.get("wer_pct")
        wer_np = lv.get("wer_no_punct_pct")
        wer_str = f"{wer_pct:.2f}%" if wer_pct is not None else "—"
        wer_np_str = f"{wer_np:.2f}%" if wer_np is not None else "—"
        print(
            f"  streams={lv['concurrent_streams']}: "
            f"avg_rtf={lv['avg_rtf']:.4f} "
            f"p50={lv['p50_rtf']:.4f} "
            f"p95={lv['p95_rtf']:.4f} "
            f"gpu_avg={lv.get('gpu_util_avg_pct')}% "
            f"gpu_peak={lv.get('gpu_util_max_pct')}% "
            f"vram_avg={lv.get('vram_used_avg_mib')} MiB "
            f"vram_peak={lv.get('vram_used_max_mib')} MiB "
            f"throughput={lv['throughput_audio_sec_per_sec']:.2f} audio-s/s "
            f"({lv['throughput_utt_per_sec']:.2f} utt/s) "
            f"wer={wer_str} wer_no_punct={wer_np_str} "
            f"(wer_n={lv.get('wer_n')}, fixed set size={result.get('wer_set_size')})"
        )
