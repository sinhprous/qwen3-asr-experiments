#!/usr/bin/env python3
"""Baseline inference benchmark for original Qwen3-ASR-1.7B on Modal.

Records hardware/software environment and RTF / throughput / GPU metrics
across representative audio duration buckets (2–5s, 5–15s, 15–30s).

Prerequisites
-------------
1) Prepared data uploaded to the Modal volume (same as training)::

    export MODAL_PROFILE=...   # or: modal profile activate <your-profile>
    modal run train_modal.py --action upload --local-dir ./data_ms_en

2) Run the benchmark::

    modal run bench_modal.py
    modal run bench_modal.py --per-bucket 25 --batch-size 1

Outputs (on the volume + printed locally)
-----------------------------------------
  /vol/checkpoints/baseline_bench/inference_baseline.json
  /vol/checkpoints/baseline_bench/inference_baseline_summary.json

RTF = inference processing time / audio duration  (lower is better).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import modal

# Local-only; Modal remote containers do not need .env.
load_dotenv(Path(__file__).resolve().parent / ".env")
MODAL_PROFILE = os.environ.get("MODAL_PROFILE", "").strip()

APP_NAME = "qwen3-asr-ms-en-bench"
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
    print(f"[write] {path}")


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
    """Hardware + software environment for the inference baseline."""
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
    if not cpu or cpu == "unknown":
        try:
            out = subprocess.check_output(["lscpu"], text=True)
            for line in out.splitlines():
                if line.lower().startswith("model name:"):
                    cpu = line.split(":", 1)[1].strip()
                    break
                if line.lower().startswith("bios model name:"):
                    cpu = line.split(":", 1)[1].strip()
        except Exception:
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


def _select_benchmark_samples(
    jsonl_paths: list[str],
    data_root: str,
    per_bucket: int = 25,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Pick representative clips across duration buckets from prepared JSONL."""
    import os
    import random

    buckets = {
        "2-5s": [],
        "5-15s": [],
        "15-30s": [],
    }
    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            continue
        for row in _load_jsonl(jsonl_path):
            path = _resolve_audio_path(row["audio"], data_root)
            if not os.path.exists(path):
                continue
            dur = _audio_duration_sec(path)
            item = {
                "audio": row["audio"],
                "path": path,
                "duration_sec": dur,
                "source_jsonl": os.path.basename(jsonl_path),
            }
            if 2.0 <= dur < 5.0:
                buckets["2-5s"].append(item)
            elif 5.0 <= dur < 15.0:
                buckets["5-15s"].append(item)
            elif 15.0 <= dur <= 30.0:
                buckets["15-30s"].append(item)

    rng = random.Random(seed)
    selected: dict[str, list[dict]] = {}
    for name, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        selected[name] = items[:per_bucket]
        print(
            f"[bench] bucket {name}: available={len(items)} "
            f"selected={len(selected[name])}"
        )
    return selected


def _summarize_rtfs(rtfs: list[float], durations: list[float], times: list[float]) -> dict:
    total_audio = float(sum(durations))
    total_proc = float(sum(times))
    return {
        "n": len(rtfs),
        "avg_rtf": float(sum(rtfs) / len(rtfs)) if rtfs else float("nan"),
        "p50_rtf": _percentile(rtfs, 50),
        "p95_rtf": _percentile(rtfs, 95),
        "min_rtf": float(min(rtfs)) if rtfs else float("nan"),
        "max_rtf": float(max(rtfs)) if rtfs else float("nan"),
        "total_audio_sec": total_audio,
        "total_proc_sec": total_proc,
        "aggregate_rtf": (total_proc / total_audio) if total_audio > 0 else float("nan"),
        "throughput_audio_sec_per_sec": (
            total_audio / total_proc if total_proc > 0 else float("nan")
        ),
        "throughput_utt_per_sec": (
            len(times) / total_proc if total_proc > 0 else float("nan")
        ),
    }


def _benchmark_inference(
    model_path: str,
    samples_by_bucket: dict[str, list[dict]],
    dtype,
    device: str = "cuda:0",
    warmup: int = 3,
    batch_size: int = 1,
) -> dict:
    """Time original-model transcription; report RTF / GPU / throughput."""
    import subprocess
    import threading
    import time

    import torch
    from qwen_asr import Qwen3ASRModel

    precision = "bfloat16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
    env = _collect_env(
        precision=precision,
        framework="qwen-asr (Qwen3ASRModel.transcribe)",
    )
    print("[bench] environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")

    print(f"[bench] loading model {model_path} …")
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
    print(f"[bench] model loaded in {load_sec:.2f}s")

    flat = [s for bucket in samples_by_bucket.values() for s in bucket]
    if not flat:
        raise RuntimeError("No benchmark samples selected")

    warm_paths = [s["path"] for s in flat[:warmup]]
    print(f"[bench] warmup n={len(warm_paths)}")
    for p in warm_paths:
        _ = model.transcribe(audio=[p])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    util_samples: list[float] = []
    mem_used_mib: list[float] = []
    stop_poll = threading.Event()

    def _poll_gpu():
        while not stop_poll.is_set():
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
            stop_poll.wait(0.25)

    poller = threading.Thread(target=_poll_gpu, daemon=True)
    if torch.cuda.is_available():
        poller.start()

    per_utt: list[dict] = []
    by_bucket: dict[str, dict] = {}

    try:
        for bucket_name, items in samples_by_bucket.items():
            if not items:
                by_bucket[bucket_name] = {
                    "n": 0,
                    "note": "no samples available in this duration range",
                }
                continue

            rtfs: list[float] = []
            durs: list[float] = []
            times: list[float] = []

            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                paths = [s["path"] for s in batch]
                audio_dur = float(sum(s["duration_sec"] for s in batch))

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                results = model.transcribe(audio=paths)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                proc = time.perf_counter() - t0

                # Per-utterance RTF: split batch wall time by audio share
                for s, res in zip(batch, results):
                    share = (
                        (s["duration_sec"] / audio_dur) * proc
                        if audio_dur > 0
                        else proc / len(batch)
                    )
                    rtf = share / s["duration_sec"] if s["duration_sec"] > 0 else float("nan")
                    rtfs.append(rtf)
                    durs.append(s["duration_sec"])
                    times.append(share)
                    per_utt.append(
                        {
                            "bucket": bucket_name,
                            "audio": s["audio"],
                            "duration_sec": s["duration_sec"],
                            "proc_sec": share,
                            "rtf": rtf,
                            "language": getattr(res, "language", "") or "",
                            "hyp_chars": len(getattr(res, "text", "") or ""),
                        }
                    )

            by_bucket[bucket_name] = _summarize_rtfs(rtfs, durs, times)
            print(
                f"[bench] {bucket_name}: n={by_bucket[bucket_name]['n']} "
                f"avg_rtf={by_bucket[bucket_name]['avg_rtf']:.4f} "
                f"p50={by_bucket[bucket_name]['p50_rtf']:.4f} "
                f"p95={by_bucket[bucket_name]['p95_rtf']:.4f} "
                f"throughput={by_bucket[bucket_name]['throughput_audio_sec_per_sec']:.2f}x"
            )
    finally:
        stop_poll.set()
        if poller.is_alive():
            poller.join(timeout=2.0)

    all_rtfs = [u["rtf"] for u in per_utt]
    all_durs = [u["duration_sec"] for u in per_utt]
    all_times = [u["proc_sec"] for u in per_utt]
    overall = _summarize_rtfs(all_rtfs, all_durs, all_times)

    peak_alloc_gb = None
    peak_reserved_gb = None
    if torch.cuda.is_available():
        peak_alloc_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        peak_reserved_gb = round(torch.cuda.max_memory_reserved() / (1024**3), 3)

    gpu_stats = {
        "util_avg_pct": (
            float(sum(util_samples) / len(util_samples)) if util_samples else None
        ),
        "util_p50_pct": _percentile(util_samples, 50) if util_samples else None,
        "util_p95_pct": _percentile(util_samples, 95) if util_samples else None,
        "util_max_pct": float(max(util_samples)) if util_samples else None,
        "util_samples": len(util_samples),
        "memory_used_avg_mib": (
            float(sum(mem_used_mib) / len(mem_used_mib)) if mem_used_mib else None
        ),
        "memory_used_max_mib": float(max(mem_used_mib)) if mem_used_mib else None,
        "torch_peak_allocated_gb": peak_alloc_gb,
        "torch_peak_reserved_gb": peak_reserved_gb,
    }

    print(
        f"[bench] overall: n={overall['n']} avg_rtf={overall['avg_rtf']:.4f} "
        f"p50={overall['p50_rtf']:.4f} p95={overall['p95_rtf']:.4f} "
        f"throughput={overall['throughput_audio_sec_per_sec']:.2f} audio-sec/sec"
    )
    print(
        f"[bench] gpu util avg={gpu_stats['util_avg_pct']} "
        f"peak_alloc_gb={peak_alloc_gb} peak_reserved_gb={peak_reserved_gb}"
    )

    return {
        "model": model_path,
        "batch_size": batch_size,
        "warmup": warmup,
        "model_load_sec": load_sec,
        "environment": env,
        "overall": overall,
        "by_bucket": by_bucket,
        "gpu": gpu_stats,
        "per_utt": per_utt,
    }


# ── Modal remote function ────────────────────────────────────────────────────


@app.function(
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60 * 2,
    memory=32768,
)
def run_benchmark(
    per_bucket: int = 25,
    batch_size: int = 1,
    warmup: int = 3,
    seed: int = 42,
    output_name: str = "baseline_bench",
) -> dict:
    """Baseline inference benchmark for original Qwen3-ASR-1.7B."""
    import os

    import torch

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", f"{CKPT_ROOT}/hf_cache")
    os.makedirs(CKPT_ROOT, exist_ok=True)

    train_src = f"{DATA_ROOT}/train.jsonl"
    dev_src = f"{DATA_ROOT}/dev.jsonl"
    test_src = f"{DATA_ROOT}/test.jsonl"
    for p in (train_src, test_src):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Upload prepared data first:\n"
                f"  export MODAL_PROFILE=<your-profile>\n"
                f"  modal run train_modal.py --action upload --local-dir ./data_ms_en"
            )

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    samples = _select_benchmark_samples(
        jsonl_paths=[test_src, dev_src, train_src],
        data_root=DATA_ROOT,
        per_bucket=per_bucket,
        seed=seed,
    )
    report = _benchmark_inference(
        model_path=MODEL_ID,
        samples_by_bucket=samples,
        dtype=dtype,
        device="cuda:0",
        warmup=warmup,
        batch_size=batch_size,
    )

    out_dir = f"{CKPT_ROOT}/{output_name}"
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "inference_baseline.json"), report)
    summary = {k: v for k, v in report.items() if k != "per_utt"}
    summary["per_utt_n"] = len(report.get("per_utt") or [])
    _write_json(os.path.join(out_dir, "inference_baseline_summary.json"), summary)
    vol.commit()
    print(f"[done] baseline → {os.path.join(out_dir, 'inference_baseline.json')}")
    return summary


# ── Local entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    per_bucket: int = 25,
    batch_size: int = 1,
    warmup: int = 3,
    seed: int = 42,
    output_name: str = "baseline_bench",
):
    """Run original Qwen3-ASR-1.7B inference baseline on Modal (L4)."""
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

    result = run_benchmark.remote(
        per_bucket=per_bucket,
        batch_size=batch_size,
        warmup=warmup,
        seed=seed,
        output_name=output_name,
    )

    # Mirror summary locally for convenience
    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "inference_baseline_summary.json",
    )
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[local] wrote {local_path}")

    print("\n=== Environment ===")
    env = result.get("environment") or {}
    for k in (
        "gpu_model",
        "gpu_memory_gb",
        "cpu",
        "ram_gb",
        "cuda_version",
        "pytorch_version",
        "inference_framework",
        "precision",
    ):
        print(f"  {k}: {env.get(k)}")

    print("\n=== Overall ===")
    overall = result.get("overall") or {}
    for k in (
        "n",
        "avg_rtf",
        "p50_rtf",
        "p95_rtf",
        "throughput_audio_sec_per_sec",
        "throughput_utt_per_sec",
    ):
        print(f"  {k}: {overall.get(k)}")

    print("\n=== By duration bucket ===")
    for name, stats in (result.get("by_bucket") or {}).items():
        if stats.get("n", 0) == 0:
            print(f"  {name}: {stats.get('note', 'empty')}")
            continue
        print(
            f"  {name}: n={stats['n']} "
            f"avg_rtf={stats['avg_rtf']:.4f} "
            f"p50={stats['p50_rtf']:.4f} "
            f"p95={stats['p95_rtf']:.4f} "
            f"throughput={stats['throughput_audio_sec_per_sec']:.2f} audio-sec/sec"
        )

    print("\n=== GPU ===")
    gpu = result.get("gpu") or {}
    for k in (
        "util_avg_pct",
        "util_p50_pct",
        "util_p95_pct",
        "util_max_pct",
        "memory_used_avg_mib",
        "memory_used_max_mib",
        "torch_peak_allocated_gb",
        "torch_peak_reserved_gb",
    ):
        print(f"  {k}: {gpu.get(k)}")

    print("\nFull summary:")
    print(json.dumps(result, indent=2))
