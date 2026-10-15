#!/usr/bin/env python3
"""Concurrent-stream load test for Qwen3-ASR-1.7B on Modal.

Backends
--------
* ``pytorch`` (default) — ``qwen-asr`` / Transformers bf16 on L4
* ``vllm`` — ``Qwen3ASRModel.LLM`` (vLLM continuous batching)
* ``transcribe_cpp`` — DISABLED for now (CUDA image cmake build is too slow;
  re-enable ``transcribe_cpp_image`` / ``run_loadtest_transcribe_cpp`` when
  prebuilt CUDA wheels exist)

Ladder: 1 → 2 → 4 → … → max_concurrency (stop on OOM).

Per level: Concurrent Streams | Avg/P50/P95 RTF | GPU avg/peak |
VRAM avg/peak | Throughput | WER | WER no-punct

Throughput / RTF / GPU use ``--waves``.
WER uses a fixed ``--wer-set-size`` clip list (identical every level).
Pass ``--skip-wer`` for throughput/GPU only.
By default ``transcribe`` gets file paths. Pass ``--preload-audio`` to
load wavs into RAM before timed waves.

Examples
--------
::

    modal run loadtest_modal.py --backend pytorch --skip-wer
    modal run loadtest_modal.py --backend vllm --skip-wer
    modal run loadtest_modal.py --backend vllm --gpu-memory-utilization 0.9 \\
        --max-model-len 8192 --max-new-tokens 512 \\
        --waves 4 --wer-set-size 128 --max-concurrency 2048
    # MS-EN ModelOpt checkpoints (quantize_modal.py → volume):
    modal run loadtest_modal.py --backend vllm --quantization fp8 \\
        --wer-set-size 128 --max-concurrency 2048
    modal run loadtest_modal.py --backend vllm --quantization nvfp4 --skip-wer
    # Optional: preload wavs to RAM before timed waves:
    # modal run loadtest_modal.py --backend vllm --preload-audio --skip-wer
    modal run loadtest_modal.py --cpu-prep-profile --backend vllm --profile-streams 256
    # Single-wave Torch profiler (vLLM) → chrome traces on volume:
    modal run loadtest_modal.py --torch-profile --backend vllm --profile-streams 256
    # modal run loadtest_modal.py --backend transcribe_cpp --gguf-quant Q8_0 --skip-wer
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
# vLLM export dirs from quantize_modal.py (ModelOpt PTQ on MS-EN train.jsonl).
QUANT_VLLM_FP8_DIR = f"{CKPT_ROOT}/qwen3asr_ms_en_fp8_vllm"
QUANT_VLLM_NVFP4_DIR = f"{CKPT_ROOT}/qwen3asr_ms_en_nvfp4_vllm"
GGUF_REPO = "handy-computer/Qwen3-ASR-1.7B-gguf"
GGUF_FILES = {
    "BF16": "Qwen3-ASR-1.7B-BF16.gguf",
    "F16": "Qwen3-ASR-1.7B-F16.gguf",
    "Q8_0": "Qwen3-ASR-1.7B-Q8_0.gguf",
    "Q6_K": "Qwen3-ASR-1.7B-Q6_K.gguf",
    "Q5_K_M": "Qwen3-ASR-1.7B-Q5_K_M.gguf",
    "Q4_K_M": "Qwen3-ASR-1.7B-Q4_K_M.gguf",
}
GPU = "L4"
TRANSCRIBE_CPP_SRC = "/opt/transcribe.cpp"
TRANSCRIBE_CPP_LIB = "/opt/transcribe.cpp/install/lib/libtranscribe.so"

# ── Images ───────────────────────────────────────────────────────────────────

pytorch_image = (
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
        "huggingface_hub",
    )
)

# CUDA wheels for transcribe-cpp are not published yet — building ggml+CUDA from
# source on Modal takes a long time. Commented out so pytorch/vllm runs stay fast.
# Re-enable this image + run_loadtest_transcribe_cpp when cu12 wheels exist.
#
# transcribe_cpp_image = (
#     modal.Image.from_registry(
#         "nvidia/cuda:12.4.1-devel-ubuntu22.04",
#         add_python="3.11",
#     )
#     .apt_install(
#         "ffmpeg",
#         "libsndfile1",
#         "git",
#         "cmake",
#         "ninja-build",
#         "build-essential",
#         "libopenblas-dev",
#         "pkg-config",
#     )
#     .run_commands(
#         f"git clone --depth 1 https://github.com/handy-computer/transcribe.cpp.git {TRANSCRIBE_CPP_SRC}",
#         f"cmake -B {TRANSCRIBE_CPP_SRC}/build -S {TRANSCRIBE_CPP_SRC} "
#         f"-G Ninja -DCMAKE_BUILD_TYPE=Release "
#         f"-DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_CUDA=ON "
#         f"-DTRANSCRIBE_INSTALL=ON -DTRANSCRIBE_BUILD_TESTS=OFF "
#         f"-DTRANSCRIBE_BUILD_EXAMPLES=OFF "
#         f"-DCMAKE_INSTALL_PREFIX={TRANSCRIBE_CPP_SRC}/install",
#         f"cmake --build {TRANSCRIBE_CPP_SRC}/build -j$(nproc)",
#         f"cmake --install {TRANSCRIBE_CPP_SRC}/build",
#         f"bash -lc 'if [ ! -f {TRANSCRIBE_CPP_LIB} ]; then "
#         f"  found=$(find {TRANSCRIBE_CPP_SRC}/build {TRANSCRIBE_CPP_SRC}/install "
#         f"    -name \"libtranscribe.so\" -print -quit); "
#         f"  test -n \"$found\" && mkdir -p $(dirname {TRANSCRIBE_CPP_LIB}) "
#         f"    && cp -L \"$found\" {TRANSCRIBE_CPP_LIB}; "
#         f"fi; ls -la {TRANSCRIBE_CPP_LIB}; test -f {TRANSCRIBE_CPP_LIB}'",
#     )
#     .pip_install(
#         "transcribe-cpp==0.4.0",
#         "numpy",
#         "soundfile",
#         "jiwer",
#         "python-dotenv",
#         "huggingface_hub",
#         "tqdm",
#     )
#     .env(
#         {
#             "TRANSCRIBE_LIBRARY": TRANSCRIBE_CPP_LIB,
#             "TRANSCRIBE_BACKEND": "cuda",
#         }
#     )
# )

# Separate image: qwen-asr[vllm] pulls vLLM + CUDA torch (heavy, but prebuilt wheels).
# nvidia-modelopt required for --quantization fp8|nvfp4 (ModelOpt vLLM checkpoints).
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "qwen-asr[vllm]",
        "nvidia-modelopt",
        "soundfile",
        "jiwer",
        "numpy",
        "tqdm",
        "python-dotenv",
        "huggingface_hub",
    )
    .env(
        {
            # Avoid fork issues with CUDA + vLLM workers inside Modal.
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# ── Helpers (run inside Modal containers) ────────────────────────────────────


def _log(msg: str) -> None:
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


def _collect_env(precision: str, framework: str, extra: dict | None = None) -> dict:
    import os
    import platform
    import subprocess

    gpu_name = None
    gpu_mem_total_gb = None
    cuda_version = None
    pytorch_version = None
    device = "cpu"

    try:
        import torch

        pytorch_version = torch.__version__
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_mem_total_gb = round(props.total_memory / (1024**3), 2)
            cuda_version = torch.version.cuda
            device = "cuda:0"
    except Exception:
        pass

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
            if len(parts) >= 2 and gpu_mem_total_gb is None:
                try:
                    gpu_mem_total_gb = round(float(parts[1]) / 1024.0, 2)
                except ValueError:
                    pass
            if len(parts) >= 3:
                driver_version = parts[2]
            device = "cuda:0"
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

    env = {
        "gpu_model": gpu_name,
        "gpu_memory_gb": gpu_mem_total_gb,
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "cuda_version": cuda_version,
        "nvidia_driver": driver_version,
        "pytorch_version": pytorch_version,
        "inference_framework": framework,
        "precision": precision,
        "device": device,
        "modal_gpu_request": GPU,
    }
    if extra:
        env.update(extra)
    return env


def _audio_duration_sec(path: str) -> float:
    import soundfile as sf

    return float(sf.info(path).duration)


def _load_wav_tuple(path: str) -> tuple:
    """Read one file into (float32 waveform, sample_rate) for qwen-asr."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return (np.ascontiguousarray(audio, dtype=np.float32), int(sr))


def _load_audio_tuples(paths: list[str]) -> list[tuple]:
    """Read files into (float32 waveform, sample_rate) tuples for qwen-asr."""
    return [_load_wav_tuple(p) for p in paths]


def _audio_for_transcribe(sample: dict):
    """Prefer preloaded (wav, sr); fall back to path string."""
    wav = sample.get("wav")
    if wav is not None:
        return wav
    return sample["path"]


def _preload_pool_audio(pool: list[dict]) -> dict:
    """Attach ``wav=(ndarray, sr)`` to each sample (shared cache by path)."""
    import time

    t0 = time.perf_counter()
    cache: dict[str, tuple] = {}
    bytes_est = 0
    for s in pool:
        path = s["path"]
        if path not in cache:
            cache[path] = _load_wav_tuple(path)
            bytes_est += int(cache[path][0].nbytes)
        s["wav"] = cache[path]
    sec = time.perf_counter() - t0
    _log(
        f"[loadtest] stage: preload audio n={len(pool)} unique={len(cache)} "
        f"~{bytes_est / (1024 ** 2):.1f} MiB in {sec:.2f}s"
    )
    return {
        "preload_audio": True,
        "preload_n": len(pool),
        "preload_unique": len(cache),
        "preload_mib": bytes_est / (1024 ** 2),
        "preload_sec": sec,
    }


def _run_cpu_prep_profile(
    engine,
    pool: list[dict],
    *,
    streams: int,
    warmup: int,
    repeats: int,
) -> dict:
    """Compare transcribe(paths) vs preloaded numpy; isolate disk decode time."""
    import time

    import torch

    concurrency = min(streams, len(pool))
    if concurrency < 1:
        raise ValueError("profile streams must be >= 1")

    reps = max(1, repeats)
    engine.warmup(pool, warmup)

    batch = pool[:concurrency]
    paths = [s["path"] for s in batch]
    audio_dur = float(sum(s["duration_sec"] for s in batch))

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _transcribe_paths() -> float:
        _sync()
        t0 = time.perf_counter()
        engine.model.transcribe(audio=paths)
        _sync()
        return time.perf_counter() - t0

    def _transcribe_tuples(tuples: list[tuple]) -> float:
        _sync()
        t0 = time.perf_counter()
        engine.model.transcribe(audio=tuples)
        _sync()
        return time.perf_counter() - t0

    _log(
        f"[cpu_prep_profile] streams={concurrency} audio={audio_dur:.1f}s "
        f"repeats={reps} (warmup={warmup} already done)"
    )

    path_secs: list[float] = []
    for i in range(reps):
        sec = _transcribe_paths()
        path_secs.append(sec)
        _log(f"[cpu_prep_profile]   transcribe(paths) repeat {i + 1}/{reps}: {sec:.3f}s")

    t_load0 = time.perf_counter()
    tuples = _load_audio_tuples(paths)
    load_sec_cold = time.perf_counter() - t_load0
    _log(f"[cpu_prep_profile]   disk load (sf.read) cold: {load_sec_cold:.3f}s")

    t_load1 = time.perf_counter()
    _ = _load_audio_tuples(paths)
    load_sec_warm = time.perf_counter() - t_load1
    _log(
        f"[cpu_prep_profile]   disk load (sf.read) warm (OS cache): "
        f"{load_sec_warm:.3f}s"
    )

    numpy_secs: list[float] = []
    for i in range(reps):
        sec = _transcribe_tuples(tuples)
        numpy_secs.append(sec)
        _log(
            f"[cpu_prep_profile]   transcribe(numpy) repeat {i + 1}/{reps}: {sec:.3f}s"
        )

    path_best = min(path_secs)
    path_last = path_secs[-1]
    numpy_best = min(numpy_secs)
    numpy_last = numpy_secs[-1]
    saved = path_last - numpy_last
    saved_pct = (saved / path_last * 100.0) if path_last > 0 else float("nan")

    summary = {
        "backend": engine.name,
        "streams": concurrency,
        "total_audio_sec": audio_dur,
        "warmup_utterances": warmup,
        "repeats": reps,
        "load_disk_sec_cold": load_sec_cold,
        "load_disk_sec_warm_cache": load_sec_warm,
        "transcribe_paths_sec": path_secs,
        "transcribe_paths_sec_best": path_best,
        "transcribe_paths_sec_last": path_last,
        "transcribe_numpy_sec": numpy_secs,
        "transcribe_numpy_sec_best": numpy_best,
        "transcribe_numpy_sec_last": numpy_last,
        "path_minus_numpy_last_sec": saved,
        "path_minus_numpy_last_pct": saved_pct,
        "load_over_path_last_pct": (
            load_sec_cold / path_last * 100.0 if path_last > 0 else float("nan")
        ),
        "throughput_paths_audio_s_per_s_last": (
            audio_dur / path_last if path_last > 0 else float("nan")
        ),
        "throughput_numpy_audio_s_per_s_last": (
            audio_dur / numpy_last if numpy_last > 0 else float("nan")
        ),
    }
    _log(
        "[cpu_prep_profile] --- summary ---\n"
        f"  paths last={path_last:.3f}s ({summary['throughput_paths_audio_s_per_s_last']:.1f} audio-s/s)\n"
        f"  numpy last={numpy_last:.3f}s ({summary['throughput_numpy_audio_s_per_s_last']:.1f} audio-s/s)\n"
        f"  delta paths-numpy last={saved:.3f}s ({saved_pct:.1f}%)\n"
        f"  disk load cold={load_sec_cold:.3f}s "
        f"({summary['load_over_path_last_pct']:.1f}% of paths last)"
    )
    return summary


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


def _duration_bucket(dur: float) -> str | None:
    """Match bench_modal buckets: 2–5s, 5–15s, 15–30s."""
    if 2.0 <= dur < 5.0:
        return "2-5s"
    if 5.0 <= dur < 15.0:
        return "5-15s"
    if 15.0 <= dur <= 30.0:
        return "15-30s"
    return None


def _select_loadtest_samples(
    jsonl_paths: list[str],
    data_root: str,
    pool_size: int = 128,
    seed: int = 42,
) -> list[dict]:
    """Build a stratified audio pool across 2–5s / 5–15s / 15–30s durations."""
    import os
    import random

    buckets: dict[str, list[dict]] = {
        "2-5s": [],
        "5-15s": [],
        "15-30s": [],
    }
    bucket_names = list(buckets.keys())

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
            if "duration" in row:
                dur = float(row["duration"])
            else:
                dur = _audio_duration_sec(path)
            bname = _duration_bucket(dur)
            if bname is None:
                continue
            buckets[bname].append(
                {
                    "audio": row["audio"],
                    "path": path,
                    "duration_sec": dur,
                    "text": row.get("text") or "",
                    "source_jsonl": os.path.basename(jsonl_path),
                    "bucket": bname,
                }
            )
            kept += 1
        _log(f"[loadtest]   kept={kept} from {os.path.basename(jsonl_path)}")

    rng = random.Random(seed)
    for name in bucket_names:
        rng.shuffle(buckets[name])
        _log(f"[loadtest]   bucket {name}: available={len(buckets[name])}")

    # Aim for equal quota per bucket; leftover slots go round-robin.
    n_buckets = len(bucket_names)
    base = pool_size // n_buckets
    rem = pool_size % n_buckets
    quotas = {name: base + (1 if i < rem else 0) for i, name in enumerate(bucket_names)}

    selected: dict[str, list[dict]] = {name: [] for name in bucket_names}
    leftover: list[dict] = []
    for name in bucket_names:
        take = min(quotas[name], len(buckets[name]))
        selected[name] = buckets[name][:take]
        leftover.extend(buckets[name][take:])

    pool = [s for name in bucket_names for s in selected[name]]
    if len(pool) < pool_size and leftover:
        rng.shuffle(leftover)
        need = pool_size - len(pool)
        pool.extend(leftover[:need])

    rng.shuffle(pool)
    if not pool:
        raise RuntimeError("No load-test audio samples found in prepared data")

    counts = {name: sum(1 for s in pool if s["bucket"] == name) for name in bucket_names}
    durs = [s["duration_sec"] for s in pool]
    _log(
        f"[loadtest] audio pool: n={len(pool)} "
        f"mix={counts} "
        f"dur_min={min(durs):.1f}s dur_p50={sorted(durs)[len(durs) // 2]:.1f}s "
        f"dur_max={max(durs):.1f}s"
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

    return threading.Thread(target=_poll, daemon=True)


def _build_concurrency_ladder(max_concurrency: int) -> list[int]:
    levels: list[int] = []
    c = 1
    while c <= max_concurrency:
        levels.append(c)
        c *= 2
    if levels[-1] != max_concurrency:
        levels.append(max_concurrency)
    return levels


def _torch_sync_and_peaks():
    """Optional torch CUDA sync + peak memory (None if torch/CUDA unavailable)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return (
                round(torch.cuda.max_memory_allocated() / (1024**3), 3),
                round(torch.cuda.max_memory_reserved() / (1024**3), 3),
            )
    except Exception:
        pass
    return None, None


def _torch_reset_peaks():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ── Backend engines ──────────────────────────────────────────────────────────


class _Hyp:
    __slots__ = ("text",)

    def __init__(self, text: str = ""):
        self.text = text or ""


class PytorchEngine:
    name = "pytorch"

    def __init__(self, model_id: str, dtype, device: str = "cuda:0"):
        import time

        import torch
        from qwen_asr import Qwen3ASRModel

        self.dtype = dtype
        self.device = device
        precision = "bfloat16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        self.precision = precision
        self.framework = "qwen-asr (Qwen3ASRModel.transcribe, concurrent batch)"
        self.model_id = model_id

        _log(f"[loadtest] stage: load pytorch model {model_id} …")
        t0 = time.perf_counter()
        self.model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.load_sec = time.perf_counter() - t0
        _log(f"[loadtest] model loaded in {self.load_sec:.2f}s")

    def env_extra(self) -> dict:
        return {"attn_implementation": "sdpa", "backend": self.name}

    def warmup(self, pool: list[dict], n: int) -> None:
        import torch

        warm_n = min(n, len(pool))
        _log(f"[loadtest] stage: warmup n={warm_n}")
        for i, s in enumerate(pool[:warm_n], start=1):
            _log(f"[loadtest]   warmup {i}/{warm_n}")
            _ = self.model.transcribe(audio=[_audio_for_transcribe(s)])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _torch_reset_peaks()
        _log("[loadtest] warmup done")

    def transcribe_batch(self, items: list[dict]) -> tuple[list[_Hyp], float]:
        import time

        import torch

        audios = [_audio_for_transcribe(s) for s in items]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = self.model.transcribe(audio=audios)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        proc = time.perf_counter() - t0
        out = [_Hyp(getattr(r, "text", "") or "") for r in results]
        return out, proc

    def close(self) -> None:
        return


def _resolve_vllm_model(
    *,
    quantization: str = "",
    model_id: str = "",
) -> tuple[str, str | None, str]:
    """Return (model_id, vllm_quantization_arg_or_None, precision_label).

    quantization:
      "" / none / bf16 → Qwen BF16 baseline
      fp8 / modelopt   → volume ``qwen3asr_ms_en_fp8_vllm`` + modelopt
      nvfp4 / int4 / modelopt_fp4 → ``qwen3asr_ms_en_nvfp4_vllm`` + modelopt_fp4
    """
    q = (quantization or "").strip().lower().replace("-", "_")
    mid = (model_id or "").strip()
    if q in ("", "none", "bf16", "bfloat16"):
        return mid or MODEL_ID, None, "bfloat16"
    if q in ("fp8", "modelopt", "fp8_modelopt"):
        return mid or QUANT_VLLM_FP8_DIR, "modelopt", "fp8_modelopt"
    if q in ("nvfp4", "int4", "fp4", "modelopt_fp4", "nvfp4_modelopt"):
        return mid or QUANT_VLLM_NVFP4_DIR, "modelopt_fp4", "nvfp4_modelopt"
    raise ValueError(
        f"Unknown --quantization {quantization!r}; "
        f"use none|fp8|nvfp4 (run quantize_modal.py first)"
    )


class VllmEngine:
    """Official qwen-asr vLLM backend: Qwen3ASRModel.LLM(...)."""

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        *,
        gpu_memory_utilization: float = 0.7,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
        max_model_len: int = 8192,
        quantization: str | None = None,
        precision: str = "bfloat16",
    ):
        import time

        from qwen_asr import Qwen3ASRModel

        self.model_id = model_id
        self.precision = precision
        self.framework = "qwen-asr (Qwen3ASRModel.LLM / vLLM)"
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.quantization = quantization

        llm_kwargs: dict = {
            "model": model_id,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_inference_batch_size": max_inference_batch_size,
            "max_new_tokens": max_new_tokens,
        }
        if max_model_len > 0:
            llm_kwargs["max_model_len"] = max_model_len
        if quantization:
            llm_kwargs["quantization"] = quantization

        _log(
            f"[loadtest] stage: load vLLM model {model_id} "
            f"(gpu_memory_utilization={gpu_memory_utilization}, "
            f"max_inference_batch_size={max_inference_batch_size}, "
            f"max_new_tokens={max_new_tokens}, "
            f"max_model_len={max_model_len if max_model_len > 0 else 'model_default'}, "
            f"quantization={quantization or 'none'}) …"
        )
        t0 = time.perf_counter()
        self.model = Qwen3ASRModel.LLM(**llm_kwargs)
        self.load_sec = time.perf_counter() - t0
        _log(f"[loadtest] vLLM model loaded in {self.load_sec:.2f}s")

    def env_extra(self) -> dict:
        return {
            "backend": self.name,
            "attn_implementation": "vllm",
            "model_id": self.model_id,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "max_model_len": self.max_model_len if self.max_model_len > 0 else None,
            "quantization": self.quantization or "none",
        }

    def warmup(self, pool: list[dict], n: int) -> None:
        warm_n = min(n, len(pool))
        _log(f"[loadtest] stage: warmup n={warm_n}")
        for i, s in enumerate(pool[:warm_n], start=1):
            _log(f"[loadtest]   warmup {i}/{warm_n}")
            _ = self.model.transcribe(audio=[_audio_for_transcribe(s)])
        _torch_reset_peaks()
        _log("[loadtest] warmup done")

    def transcribe_batch(self, items: list[dict]) -> tuple[list[_Hyp], float]:
        import time

        audios = [_audio_for_transcribe(s) for s in items]
        t0 = time.perf_counter()
        results = self.model.transcribe(audio=audios)
        proc = time.perf_counter() - t0
        out = [_Hyp(getattr(r, "text", "") or "") for r in results]
        return out, proc

    def close(self) -> None:
        return


class TranscribeCppEngine:
    name = "transcribe_cpp"

    def __init__(self, gguf_path: str, prefer_backend: str = "cuda"):
        import os
        import time

        import transcribe_cpp as tcp

        self.gguf_path = gguf_path
        self.precision = os.path.basename(gguf_path)
        self.framework = "transcribe.cpp (Python bindings, ggml CUDA)"
        self._pcm_cache: dict[str, object] = {}
        self._tcp = tcp

        lib = os.environ.get("TRANSCRIBE_LIBRARY", "")
        _log(f"[loadtest] transcribe_cpp version={tcp.__version__}")
        try:
            _log(f"[loadtest]   native={tcp.native_version()} lib={tcp.library_path()}")
        except Exception as exc:
            _log(f"[loadtest]   library_path probe: {exc}")
        if lib:
            _log(f"[loadtest]   TRANSCRIBE_LIBRARY={lib}")

        backend = prefer_backend
        try:
            if prefer_backend != "auto" and hasattr(tcp, "backend_available"):
                if not tcp.backend_available(prefer_backend):
                    _log(
                        f"[loadtest] backend '{prefer_backend}' unavailable; falling back to auto"
                    )
                    backend = "auto"
        except Exception as exc:
            _log(f"[loadtest] backend_available check failed ({exc}); using auto")
            backend = "auto"

        _log(f"[loadtest] stage: load GGUF {gguf_path} (backend={backend}) …")
        t0 = time.perf_counter()
        self._model_cm = tcp.Model(gguf_path, backend=backend)
        self.model = self._model_cm.__enter__()
        self._session_cm = self.model.session()
        self.session = self._session_cm.__enter__()
        self.load_sec = time.perf_counter() - t0
        try:
            used = getattr(self.model, "backend", backend)
        except Exception:
            used = backend
        _log(f"[loadtest] model loaded in {self.load_sec:.2f}s (active backend={used})")
        self.active_backend = str(used)

    def env_extra(self) -> dict:
        import os

        return {
            "backend": self.name,
            "gguf_path": self.gguf_path,
            "transcribe_cpp_backend": getattr(self, "active_backend", None),
            "transcribe_cpp_lib": os.environ.get("TRANSCRIBE_LIBRARY"),
        }

    def _load_pcm(self, path: str):
        import numpy as np
        import soundfile as sf

        cached = self._pcm_cache.get(path)
        if cached is not None:
            return cached
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if int(sr) != 16000:
            raise ValueError(f"{path}: expected 16 kHz, got {sr}")
        pcm = np.ascontiguousarray(audio, dtype=np.float32)
        self._pcm_cache[path] = pcm
        return pcm

    def warmup(self, pool: list[dict], n: int) -> None:
        warm_n = min(n, len(pool))
        _log(f"[loadtest] stage: warmup n={warm_n}")
        for i, s in enumerate(pool[:warm_n], start=1):
            _log(f"[loadtest]   warmup {i}/{warm_n}")
            pcm = self._load_pcm(s["path"])
            _ = self.session.run(pcm)
        _log("[loadtest] warmup done")

    def transcribe_batch(self, items: list[dict]) -> tuple[list[_Hyp], float]:
        import time

        pcms = [self._load_pcm(s["path"]) for s in items]
        t0 = time.perf_counter()
        results = self.session.run_batch(pcms)
        proc = time.perf_counter() - t0
        out: list[_Hyp] = []
        for r in results:
            if r is None:
                out.append(_Hyp(""))
            else:
                out.append(_Hyp(getattr(r, "text", "") or ""))
        return out, proc

    def close(self) -> None:
        try:
            self._session_cm.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._model_cm.__exit__(None, None, None)
        except Exception:
            pass


def _ensure_gguf(quant: str) -> str:
    """Download GGUF onto the Modal volume if missing; return local path."""
    import os

    from huggingface_hub import hf_hub_download

    quant = quant.strip()
    if quant not in GGUF_FILES:
        raise ValueError(
            f"Unknown gguf quant {quant!r}; choose one of {sorted(GGUF_FILES)}"
        )
    filename = GGUF_FILES[quant]
    dest_dir = f"{CKPT_ROOT}/gguf/Qwen3-ASR-1.7B"
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        _log(f"[loadtest] GGUF cached: {dest}")
        return dest

    _log(f"[loadtest] downloading {GGUF_REPO}/{filename} …")
    path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=filename,
        local_dir=dest_dir,
    )
    _log(f"[loadtest] GGUF ready: {path}")
    return path


def _build_engine(
    backend: str,
    *,
    gguf_quant: str = "Q8_0",
    gpu_memory_utilization: float = 0.7,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 512,
    max_model_len: int = 8192,
    quantization: str = "",
    model_id: str = "",
):
    backend = backend.strip().lower().replace("-", "_")
    if backend in ("pytorch", "torch", "qwen_asr"):
        import torch

        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        mid = (model_id or "").strip() or MODEL_ID
        return PytorchEngine(mid, dtype=dtype, device="cuda:0")
    if backend in ("vllm", "llm"):
        mid, vllm_q, precision = _resolve_vllm_model(
            quantization=quantization, model_id=model_id
        )
        return VllmEngine(
            mid,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            quantization=vllm_q,
            precision=precision,
        )
    if backend in ("transcribe_cpp", "cpp", "gguf"):
        gguf_path = _ensure_gguf(gguf_quant)
        return TranscribeCppEngine(gguf_path, prefer_backend="cuda")
    raise ValueError(
        f"Unknown backend {backend!r}; use pytorch|vllm|transcribe_cpp"
    )


# ── Load-test core ───────────────────────────────────────────────────────────


def _run_wer_on_fixed_set(
    engine,
    wer_set: list[dict],
    concurrency: int,
) -> tuple[dict, list[dict], list[str]]:
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
        try:
            results, proc = engine.transcribe_batch(batch)
        except Exception as exc:
            if _is_oom(exc):
                raise
            errors.append(f"wer_batch={bi}: {type(exc).__name__}: {exc}")
            _log(f"[loadtest]   wer batch {bi + 1}/{n_batches} error: {exc}")
            continue
        _log(f"[loadtest]   wer batch {bi + 1}/{n_batches} done in {proc:.2f}s")

        for s, res in zip(batch, results):
            hyp = res.text
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
    engine,
    pool: list[dict],
    wer_set: list[dict],
    concurrency: int,
    waves: int,
    skip_wer: bool = False,
) -> dict:
    import threading
    import time

    thr_pool = pool
    if concurrency > len(thr_pool):
        reps = (concurrency // len(thr_pool)) + 1
        thr_pool = (thr_pool * reps)[: max(concurrency * waves, concurrency)]

    util_samples: list[float] = []
    mem_used_mib: list[float] = []
    stop_poll = threading.Event()
    poller = _gpu_poller(stop_poll, util_samples, mem_used_mib)
    _torch_reset_peaks()
    poller.start()

    rtfs: list[float] = []
    per_utt: list[dict] = []
    wave_times: list[float] = []
    total_audio = 0.0
    errors: list[str] = []

    wall_t0 = time.perf_counter()
    try:
        for wave_i in range(waves):
            batch = []
            for j in range(concurrency):
                batch.append(thr_pool[(wave_i * concurrency + j) % len(thr_pool)])
            audio_dur = float(sum(s["duration_sec"] for s in batch))
            total_audio += audio_dur

            _log(
                f"[loadtest]   wave {wave_i + 1}/{waves} "
                f"streams={concurrency} audio={audio_dur:.1f}s …"
            )
            try:
                results, proc = engine.transcribe_batch(batch)
            except Exception as exc:
                if _is_oom(exc):
                    raise
                errors.append(f"wave={wave_i}: {type(exc).__name__}: {exc}")
                _log(f"[loadtest]   wave {wave_i + 1} error: {exc}")
                continue
            wave_times.append(proc)
            _log(f"[loadtest]   wave {wave_i + 1}/{waves} done in {proc:.2f}s")

            for s, res in zip(batch, results):
                share = (
                    (s["duration_sec"] / audio_dur) * proc
                    if audio_dur > 0
                    else proc / len(batch)
                )
                rtf = share / s["duration_sec"] if s["duration_sec"] > 0 else float("nan")
                rtfs.append(rtf)
                per_utt.append(
                    {
                        "wave": wave_i,
                        "audio": s["audio"],
                        "duration_sec": s["duration_sec"],
                        "proc_sec": share,
                        "rtf": rtf,
                        "hyp_chars": len(res.text),
                    }
                )
    finally:
        stop_poll.set()
        if poller.is_alive():
            poller.join(timeout=2.0)

    wall = time.perf_counter() - wall_t0
    peak_alloc_gb, peak_reserved_gb = _torch_sync_and_peaks()

    throughput_audio = (total_audio / wall) if wall > 0 else float("nan")
    throughput_utt = (len(rtfs) / wall) if wall > 0 else float("nan")

    if skip_wer:
        _log("[loadtest]   wer pass: skipped (--skip-wer)")
        wer_stats = None
        wer_per_utt: list[dict] = []
    else:
        wer_stats, wer_per_utt, wer_errors = _run_wer_on_fixed_set(
            engine, wer_set, concurrency
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
                if x != x:
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
    engine,
    pool: list[dict],
    waves: int = 4,
    max_concurrency: int = 256,
    wer_set_size: int = 128,
    skip_wer: bool = False,
    warmup: int = 3,
    preload_audio: bool = False,
) -> dict:
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

    preload_stats: dict = {"preload_audio": False}
    if preload_audio and engine.name in ("pytorch", "vllm"):
        preload_stats = _preload_pool_audio(pool)
    elif preload_audio:
        _log(
            f"[loadtest] preload_audio skipped for backend={engine.name} "
            f"(path/pcm handled inside engine)"
        )
    else:
        _log("[loadtest] preload_audio=False (transcribe uses file paths)")

    env = _collect_env(
        precision=engine.precision,
        framework=engine.framework,
        extra={**engine.env_extra(), **preload_stats},
    )
    _log("[loadtest] environment:")
    for k, v in env.items():
        _log(f"  {k}: {v}")

    engine.warmup(pool, warmup)

    ladder = _build_concurrency_ladder(max_concurrency)
    _log(f"[loadtest] stage: concurrency ladder {ladder}")
    _log(f"[loadtest] waves per level (throughput): {waves}")
    if skip_wer:
        _log("[loadtest] wer: skipped")
    else:
        _log(f"[loadtest] wer_set_size (fair WER): {len(wer_set)}")

    levels: list[dict] = []
    stopped_reason = None

    try:
        for concurrency in ladder:
            _log(f"\n[loadtest] === concurrent_streams={concurrency} ===")
            try:
                result = _run_concurrency_level(
                    engine=engine,
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
                _torch_reset_peaks()
                break

            result["status"] = "ok"
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
            _torch_reset_peaks()
    finally:
        engine.close()

    table = _format_table(levels)
    _log("\n=== Load test results ===")
    _log(table)
    if stopped_reason:
        _log(f"\n[loadtest] stopped early: {stopped_reason}")

    level_summaries = []
    for lv in levels:
        compact = {
            k: v for k, v in lv.items() if k not in ("per_utt", "wer_per_utt")
        }
        level_summaries.append(compact)

    return {
        "model": getattr(engine, "model_id", None) or getattr(engine, "gguf_path", None),
        "backend": engine.name,
        "variant": f"{engine.name}_baseline",
        "warmup": warmup,
        "waves": waves,
        "skip_wer": skip_wer,
        "preload_audio": bool(preload_stats.get("preload_audio")),
        "wer_set_size": 0 if skip_wer else len(wer_set),
        "max_concurrency": max_concurrency,
        "concurrency_ladder": ladder,
        "model_load_sec": engine.load_sec,
        "environment": env,
        "pool_size": len(pool),
        "levels": level_summaries,
        "levels_detail": levels,
        "table": table,
        "stopped_reason": stopped_reason,
    }


def _run_loadtest_impl(
    *,
    backend: str,
    waves: int,
    warmup: int,
    seed: int,
    pool_size: int,
    wer_set_size: int,
    skip_wer: bool,
    max_concurrency: int,
    gguf_quant: str,
    output_name: str,
    gpu_memory_utilization: float = 0.7,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 512,
    max_model_len: int = 8192,
    preload_audio: bool = False,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    import os

    _log(f"[loadtest] stage: start remote worker (backend={backend})")
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
    )

    engine = _build_engine(
        backend,
        gguf_quant=gguf_quant,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        quantization=quantization,
        model_id=model_id,
    )
    report = _loadtest_inference(
        engine=engine,
        pool=pool,
        waves=waves,
        max_concurrency=max_concurrency,
        wer_set_size=wer_set_size,
        skip_wer=skip_wer,
        warmup=warmup,
        preload_audio=preload_audio,
    )

    if not output_name:
        if backend.startswith("transcribe"):
            tag = gguf_quant.lower()
        elif backend in ("vllm", "llm"):
            qtag = (quantization or "bf16").strip().lower() or "bf16"
            tag = (
                f"{qtag}_gmu{gpu_memory_utilization:.2f}_"
                f"mml{max_model_len}_mnt{max_new_tokens}"
            ).replace(".", "p")
        else:
            tag = "bf16"
        output_name = f"loadtest_{engine.name}_{tag}"

    _log("[loadtest] stage: write results")
    out_dir = f"{CKPT_ROOT}/{output_name}"
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "loadtest.json"), report)
    summary = {k: v for k, v in report.items() if k != "levels_detail"}
    _write_json(os.path.join(out_dir, "loadtest_summary.json"), summary)
    vol.commit()
    _log(f"[done] loadtest → {os.path.join(out_dir, 'loadtest.json')}")
    return summary


def _get_vllm_llm(engine) -> object | None:
    """Best-effort unwrap of the underlying vllm.LLM from Qwen3ASRModel."""
    asr = getattr(engine, "model", None)
    if asr is None:
        return None
    for attr in ("model", "llm", "_llm", "engine"):
        inner = getattr(asr, attr, None)
        if inner is None:
            continue
        if hasattr(inner, "start_profile") and hasattr(inner, "stop_profile"):
            return inner
        # vLLM LLM often exposes .llm_engine
        if hasattr(inner, "llm_engine") and hasattr(inner, "start_profile"):
            return inner
    if hasattr(asr, "start_profile") and hasattr(asr, "stop_profile"):
        return asr
    return None


def _list_profile_files(profile_dir: str) -> list[dict]:
    import os

    out: list[dict] = []
    if not os.path.isdir(profile_dir):
        return out
    for root, _dirs, files in os.walk(profile_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1
            out.append(
                {
                    "path": path,
                    "relpath": os.path.relpath(path, profile_dir),
                    "bytes": size,
                }
            )
    out.sort(key=lambda x: x["relpath"])
    return out


def _run_torch_profile(
    engine,
    pool: list[dict],
    *,
    streams: int,
    warmup: int,
    profile_dir: str,
) -> dict:
    """Profile one preloaded transcribe wave (vLLM start/stop or torch.profiler)."""
    import os
    import time

    import torch

    concurrency = min(streams, len(pool))
    if concurrency < 1:
        raise ValueError("profile streams must be >= 1")

    os.makedirs(profile_dir, exist_ok=True)
    batch = pool[:concurrency]
    audio_dur = float(sum(s["duration_sec"] for s in batch))

    engine.warmup(pool, warmup)

    llm = _get_vllm_llm(engine)
    method = "unknown"
    chrome_trace = os.path.join(profile_dir, "chrome_trace.json")

    _log(
        f"[torch_profile] streams={concurrency} audio={audio_dur:.1f}s "
        f"dir={profile_dir} vllm_llm={'yes' if llm is not None else 'no'}"
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    if llm is not None:
        method = "vllm.start_profile"
        _log("[torch_profile] using vLLM LLM.start_profile / stop_profile")
        llm.start_profile()
        try:
            _results, proc = engine.transcribe_batch(batch)
        finally:
            llm.stop_profile()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    else:
        method = "torch.profiler"
        _log("[torch_profile] vLLM profiler API missing; using torch.profiler")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            _results, proc = engine.transcribe_batch(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        try:
            prof.export_chrome_trace(chrome_trace)
            _log(f"[torch_profile] wrote {chrome_trace}")
        except Exception as exc:
            _log(f"[torch_profile] export_chrome_trace failed: {exc}")
            try:
                key_avg = prof.key_averages()
                table_path = os.path.join(profile_dir, "key_averages.txt")
                with open(table_path, "w", encoding="utf-8") as f:
                    f.write(key_avg.table(sort_by="cuda_time_total", row_limit=80))
                _log(f"[torch_profile] wrote {table_path}")
            except Exception as exc2:
                _log(f"[torch_profile] key_averages dump failed: {exc2}")

    wall = time.perf_counter() - t0
    files = _list_profile_files(profile_dir)
    for fmeta in files:
        _log(
            f"[torch_profile]   artifact {fmeta['relpath']} "
            f"({fmeta['bytes'] / (1024 ** 2):.2f} MiB)"
        )

    thr = (audio_dur / proc) if proc > 0 else float("nan")
    summary = {
        "method": method,
        "streams": concurrency,
        "total_audio_sec": audio_dur,
        "proc_sec": proc,
        "wall_sec": wall,
        "throughput_audio_s_per_s": thr,
        "profile_dir": profile_dir,
        "artifacts": files,
        "vllm_profiler_dir_env": os.environ.get("VLLM_TORCH_PROFILER_DIR"),
    }
    _log(
        f"[torch_profile] done method={method} proc={proc:.3f}s "
        f"throughput={thr:.1f} audio-s/s artifacts={len(files)}"
    )
    return summary


def _run_torch_profile_remote(
    *,
    backend: str,
    seed: int,
    pool_size: int,
    profile_streams: int,
    warmup: int,
    gpu_memory_utilization: float,
    max_inference_batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    gguf_quant: str,
    preload_audio: bool,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    import os
    import time

    if backend.strip().lower().replace("-", "_") not in ("vllm", "llm"):
        raise ValueError("--torch-profile currently supports --backend vllm only")

    _log("[torch_profile] start")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", f"{CKPT_ROOT}/hf_cache")
    os.makedirs(CKPT_ROOT, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"loadtest_vllm_torch_profile_{stamp}"
    out_dir = f"{CKPT_ROOT}/{out_name}"
    profile_dir = f"{out_dir}/traces"
    os.makedirs(profile_dir, exist_ok=True)

    # Must be set before constructing vLLM / Qwen3ASRModel.LLM.
    os.environ["VLLM_TORCH_PROFILER_DIR"] = profile_dir
    os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "1"

    test_src = f"{DATA_ROOT}/test.jsonl"
    dev_src = f"{DATA_ROOT}/dev.jsonl"
    train_src = f"{DATA_ROOT}/train.jsonl"
    for p in (train_src, test_src):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")

    need_pool = max(pool_size, profile_streams)
    pool = _select_loadtest_samples(
        jsonl_paths=[test_src, dev_src, train_src],
        data_root=DATA_ROOT,
        pool_size=need_pool,
        seed=seed,
    )

    preload_stats: dict = {"preload_audio": False}
    if preload_audio:
        preload_stats = _preload_pool_audio(pool)

    engine = _build_engine(
        backend="vllm",
        gguf_quant=gguf_quant,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        quantization=quantization,
        model_id=model_id,
    )
    try:
        profile = _run_torch_profile(
            engine,
            pool,
            streams=profile_streams,
            warmup=warmup,
            profile_dir=profile_dir,
        )
    finally:
        engine.close()

    env = _collect_env(
        precision=engine.precision,
        framework=engine.framework,
        extra={**engine.env_extra(), **preload_stats},
    )
    report = {
        "kind": "torch_profile",
        "backend": engine.name,
        "model": getattr(engine, "model_id", None),
        "profile_streams": profile_streams,
        "output_dir": out_dir,
        "environment": env,
        "profile": profile,
    }
    _write_json(os.path.join(out_dir, "torch_profile_summary.json"), report)
    vol.commit()
    _log(f"[done] torch profile → {out_dir}")
    return report


def _run_cpu_prep_profile_remote(
    *,
    backend: str,
    seed: int,
    pool_size: int,
    profile_streams: int,
    warmup: int,
    profile_repeats: int,
    gpu_memory_utilization: float,
    max_inference_batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    gguf_quant: str,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    import os

    _log(f"[cpu_prep_profile] start (backend={backend})")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", f"{CKPT_ROOT}/hf_cache")
    os.makedirs(CKPT_ROOT, exist_ok=True)

    test_src = f"{DATA_ROOT}/test.jsonl"
    dev_src = f"{DATA_ROOT}/dev.jsonl"
    train_src = f"{DATA_ROOT}/train.jsonl"
    for p in (train_src, test_src):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")

    need_pool = max(pool_size, profile_streams)
    pool = _select_loadtest_samples(
        jsonl_paths=[test_src, dev_src, train_src],
        data_root=DATA_ROOT,
        pool_size=need_pool,
        seed=seed,
    )
    engine = _build_engine(
        backend,
        gguf_quant=gguf_quant,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        quantization=quantization,
        model_id=model_id,
    )
    try:
        profile = _run_cpu_prep_profile(
            engine,
            pool,
            streams=profile_streams,
            warmup=warmup,
            repeats=profile_repeats,
        )
    finally:
        engine.close()

    env = _collect_env(
        precision=engine.precision,
        framework=engine.framework,
        extra=engine.env_extra(),
    )
    return {
        "kind": "cpu_prep_profile",
        "backend": engine.name,
        "model": getattr(engine, "model_id", None),
        "profile_streams": profile_streams,
        "environment": env,
        "profile": profile,
    }


# ── Modal remote functions ───────────────────────────────────────────────────


@app.function(
    image=pytorch_image,
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60 * 4,
    memory=32768,
)
def run_loadtest_pytorch(
    waves: int = 4,
    warmup: int = 3,
    seed: int = 42,
    pool_size: int = 128,
    wer_set_size: int = 128,
    skip_wer: bool = False,
    max_concurrency: int = 256,
    gguf_quant: str = "Q8_0",
    output_name: str = "",
    gpu_memory_utilization: float = 0.7,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 512,
    max_model_len: int = 8192,
    preload_audio: bool = False,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    return _run_loadtest_impl(
        backend="pytorch",
        waves=waves,
        warmup=warmup,
        seed=seed,
        pool_size=pool_size,
        wer_set_size=wer_set_size,
        skip_wer=skip_wer,
        max_concurrency=max_concurrency,
        gguf_quant=gguf_quant,
        output_name=output_name or "loadtest_pytorch_bf16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        preload_audio=preload_audio,
        quantization=quantization,
        model_id=model_id,
    )


@app.function(
    image=vllm_image,
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60 * 4,
    memory=65536,
)
def run_loadtest_vllm(
    waves: int = 4,
    warmup: int = 3,
    seed: int = 42,
    pool_size: int = 128,
    wer_set_size: int = 128,
    skip_wer: bool = False,
    max_concurrency: int = 256,
    gguf_quant: str = "Q8_0",
    output_name: str = "",
    gpu_memory_utilization: float = 0.9,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 512,
    max_model_len: int = 8192,
    preload_audio: bool = False,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    """vLLM backend via Qwen3ASRModel.LLM (separate heavy image)."""
    return _run_loadtest_impl(
        backend="vllm",
        waves=waves,
        warmup=warmup,
        seed=seed,
        pool_size=pool_size,
        wer_set_size=wer_set_size,
        skip_wer=skip_wer,
        max_concurrency=max_concurrency,
        gguf_quant=gguf_quant,
        output_name=output_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        preload_audio=preload_audio,
        quantization=quantization,
        model_id=model_id,
    )


@app.function(
    image=pytorch_image,
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60,
    memory=32768,
)
def run_cpu_prep_profile_pytorch(
    seed: int = 42,
    pool_size: int = 128,
    profile_streams: int = 256,
    warmup: int = 3,
    profile_repeats: int = 2,
    gguf_quant: str = "Q8_0",
    gpu_memory_utilization: float = 0.7,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 512,
    max_model_len: int = 8192,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    return _run_cpu_prep_profile_remote(
        backend="pytorch",
        seed=seed,
        pool_size=pool_size,
        profile_streams=profile_streams,
        warmup=warmup,
        profile_repeats=profile_repeats,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        gguf_quant=gguf_quant,
        quantization=quantization,
        model_id=model_id,
    )


@app.function(
    image=vllm_image,
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60,
    memory=65536,
)
def run_cpu_prep_profile_vllm(
    seed: int = 42,
    pool_size: int = 128,
    profile_streams: int = 256,
    warmup: int = 3,
    profile_repeats: int = 2,
    gguf_quant: str = "Q8_0",
    gpu_memory_utilization: float = 0.9,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 512,
    max_model_len: int = 8192,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    return _run_cpu_prep_profile_remote(
        backend="vllm",
        seed=seed,
        pool_size=pool_size,
        profile_streams=profile_streams,
        warmup=warmup,
        profile_repeats=profile_repeats,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        gguf_quant=gguf_quant,
        quantization=quantization,
        model_id=model_id,
    )


@app.function(
    image=vllm_image,
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60,
    memory=65536,
)
def run_torch_profile_vllm(
    seed: int = 42,
    pool_size: int = 128,
    profile_streams: int = 256,
    warmup: int = 2,
    gguf_quant: str = "Q8_0",
    gpu_memory_utilization: float = 0.9,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 512,
    max_model_len: int = 8192,
    preload_audio: bool = False,
    quantization: str = "",
    model_id: str = "",
) -> dict:
    """One-wave vLLM Torch profiler; writes chrome traces under checkpoints/."""
    return _run_torch_profile_remote(
        backend="vllm",
        seed=seed,
        pool_size=pool_size,
        profile_streams=profile_streams,
        warmup=warmup,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        gguf_quant=gguf_quant,
        preload_audio=preload_audio,
        quantization=quantization,
        model_id=model_id,
    )


# Disabled with transcribe_cpp_image (slow cmake CUDA build). Uncomment both together.
#
# @app.function(
#     image=transcribe_cpp_image,
#     volumes={VOL_ROOT: vol},
#     gpu=GPU,
#     timeout=60 * 60 * 4,
#     memory=32768,
# )
# def run_loadtest_transcribe_cpp(
#     waves: int = 4,
#     warmup: int = 3,
#     seed: int = 42,
#     pool_size: int = 128,
#     wer_set_size: int = 128,
#     skip_wer: bool = False,
#     max_concurrency: int = 256,
#     gguf_quant: str = "Q8_0",
#     output_name: str = "",
#     gpu_memory_utilization: float = 0.7,
#     max_inference_batch_size: int = -1,
#     max_new_tokens: int = 4096,
# ) -> dict:
#     return _run_loadtest_impl(
#         backend="transcribe_cpp",
#         waves=waves,
#         warmup=warmup,
#         seed=seed,
#         pool_size=pool_size,
#         wer_set_size=wer_set_size,
#         skip_wer=skip_wer,
#         max_concurrency=max_concurrency,
#         gguf_quant=gguf_quant,
#         output_name=output_name
#         or f"loadtest_transcribe_cpp_{gguf_quant.lower()}",
#         gpu_memory_utilization=gpu_memory_utilization,
#         max_inference_batch_size=max_inference_batch_size,
#         max_new_tokens=max_new_tokens,
#     )


# ── Local entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    backend: str = "pytorch",
    waves: int = 4,
    warmup: int = 3,
    seed: int = 42,
    pool_size: int = 128,
    wer_set_size: int = 128,
    skip_wer: bool = False,
    max_concurrency: int = 256,
    gguf_quant: str = "Q8_0",
    output_name: str = "",
    gpu_memory_utilization: float = 0.9,
    max_inference_batch_size: int = -1,
    max_new_tokens: int = 8192, # 512,
    max_model_len: int = 65536, # 8192,
    cpu_prep_profile: bool = False,
    torch_profile: bool = False,
    profile_streams: int = 256,
    profile_repeats: int = 2,
    preload_audio: bool = False,
    quantization: str = "",
    model_id: str = "",
):
    """Run concurrent ASR load test on Modal (L4).

    backend: pytorch | vllm  (transcribe_cpp disabled — slow CUDA image build)
    gguf_quant: BF16 | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M  (cpp only, when re-enabled)
    gpu_memory_utilization / max_inference_batch_size / max_new_tokens /
    max_model_len: vLLM only (max_model_len=0 uses model default)
    quantization: none|fp8|nvfp4 (vLLM; MS-EN exports under /vol/checkpoints/)
    model_id: optional path/HF id override (vLLM/pytorch)
    preload_audio: load wavs to RAM before timed waves (default False; --preload-audio)
    torch_profile: one-wave vLLM Torch profiler (chrome traces on volume)
    """
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

    backend_norm = backend.strip().lower().replace("-", "_")
    profile_kwargs = dict(
        seed=seed,
        pool_size=pool_size,
        profile_streams=profile_streams,
        warmup=warmup,
        profile_repeats=profile_repeats,
        gguf_quant=gguf_quant,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        quantization=quantization,
        model_id=model_id,
    )
    kwargs = dict(
        waves=waves,
        warmup=warmup,
        seed=seed,
        pool_size=pool_size,
        wer_set_size=wer_set_size,
        skip_wer=skip_wer,
        max_concurrency=max_concurrency,
        gguf_quant=gguf_quant,
        output_name=output_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        preload_audio=preload_audio,
        quantization=quantization,
        model_id=model_id,
    )

    if torch_profile and cpu_prep_profile:
        raise SystemExit("Use only one of --torch-profile or --cpu-prep-profile")

    if torch_profile:
        if backend_norm not in ("vllm", "llm"):
            raise SystemExit("--torch-profile requires --backend vllm")
        result = run_torch_profile_vllm.remote(
            seed=seed,
            pool_size=pool_size,
            profile_streams=profile_streams,
            warmup=warmup,
            gguf_quant=gguf_quant,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            preload_audio=preload_audio,
            quantization=quantization,
            model_id=model_id,
        )
    elif cpu_prep_profile:
        if backend_norm in ("pytorch", "torch", "qwen_asr"):
            result = run_cpu_prep_profile_pytorch.remote(**profile_kwargs)
        elif backend_norm in ("vllm", "llm"):
            result = run_cpu_prep_profile_vllm.remote(**profile_kwargs)
        else:
            raise SystemExit(
                f"Unknown --backend {backend!r} for --cpu-prep-profile; use pytorch or vllm"
            )
    elif backend_norm in ("pytorch", "torch", "qwen_asr"):
        result = run_loadtest_pytorch.remote(**kwargs)
    elif backend_norm in ("vllm", "llm"):
        result = run_loadtest_vllm.remote(**kwargs)
    elif backend_norm in ("transcribe_cpp", "cpp", "gguf"):
        raise SystemExit(
            "backend=transcribe_cpp is disabled (CUDA cmake image build is too slow). "
            "Use --backend pytorch or --backend vllm. "
            "Re-enable transcribe_cpp_image + run_loadtest_transcribe_cpp in "
            "loadtest_modal.py when prebuilt CUDA wheels exist."
        )
    else:
        raise SystemExit(
            f"Unknown --backend {backend!r}; use pytorch or vllm"
        )

    if torch_profile:
        out_name = "loadtest_torch_profile_summary.json"
    elif cpu_prep_profile:
        out_name = "loadtest_cpu_prep_profile.json"
    else:
        out_name = "loadtest_baseline_summary.json"
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[local] wrote {local_path}")

    if torch_profile:
        p = result.get("profile") or {}
        print("\n=== Torch profile (vLLM) ===")
        print(f"  backend: {result.get('backend')}")
        print(f"  method: {p.get('method')}")
        print(f"  streams: {p.get('streams')}")
        print(f"  proc: {p.get('proc_sec'):.3f}s")
        print(f"  throughput: {p.get('throughput_audio_s_per_s'):.1f} audio-s/s")
        print(f"  remote dir: {result.get('output_dir')}")
        arts = p.get("artifacts") or []
        print(f"  artifacts: {len(arts)}")
        for a in arts[:20]:
            print(f"    - {a.get('relpath')} ({(a.get('bytes') or 0) / (1024 ** 2):.2f} MiB)")
        rel = (result.get("output_dir") or "").replace(f"{VOL_ROOT}/", "")
        if rel:
            print(
                "\n  Download traces:\n"
                f"    modal volume get {VOLUME_NAME} {rel} ./vllm_torch_profile/"
            )
            print("  Open chrome_trace*.json in https://ui.perfetto.dev/")
        return

    if cpu_prep_profile:
        p = result.get("profile") or {}
        print("\n=== CPU prep profile ===")
        print(f"  backend: {result.get('backend')}")
        print(f"  streams: {p.get('streams')}")
        print(f"  transcribe(paths) last: {p.get('transcribe_paths_sec_last'):.3f}s")
        print(f"  transcribe(numpy) last: {p.get('transcribe_numpy_sec_last'):.3f}s")
        print(
            f"  delta: {p.get('path_minus_numpy_last_sec'):.3f}s "
            f"({p.get('path_minus_numpy_last_pct'):.1f}%)"
        )
        print(f"  disk load cold: {p.get('load_disk_sec_cold'):.3f}s")
        print(
            f"  throughput paths: {p.get('throughput_paths_audio_s_per_s_last'):.1f} audio-s/s"
        )
        print(
            f"  throughput numpy: {p.get('throughput_numpy_audio_s_per_s_last'):.1f} audio-s/s"
        )
        return

    print("\n=== Environment ===")
    env = result.get("environment") or {}
    for k in (
        "backend",
        "gpu_model",
        "gpu_memory_gb",
        "cuda_version",
        "pytorch_version",
        "inference_framework",
        "precision",
        "gguf_path",
        "transcribe_cpp_backend",
        "gpu_memory_utilization",
        "max_inference_batch_size",
        "max_new_tokens",
        "max_model_len",
        "quantization",
        "model_id",
        "preload_audio",
        "preload_unique",
        "preload_mib",
        "preload_sec",
    ):
        if k in env and env.get(k) is not None:
            print(f"  {k}: {env.get(k)}")

    print(f"\n=== Concurrent stream ladder ({result.get('backend')}) ===")
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
