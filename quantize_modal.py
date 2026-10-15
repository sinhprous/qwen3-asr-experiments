#!/usr/bin/env python3
"""ModelOpt FP8 / NVFP4 quantization for Qwen3-ASR on Modal (MS-EN calibration).

Workflow
--------
1) Prep + upload data (same as training)::

    python prep_data.py --src ... --out ./data_ms_en
    modal run train_modal.py --action upload --local-dir ./data_ms_en

2) Quantize + export vLLM checkpoints on the shared volume::

    modal run quantize_modal.py --format fp8
    modal run quantize_modal.py --format nvfp4 --algorithm max

3) Load-test with exported checkpoints::

    modal run loadtest_modal.py --backend vllm --quantization fp8
    modal run loadtest_modal.py --backend vllm --quantization nvfp4

Optional LoRA merge before PTQ::

    modal run quantize_modal.py --format fp8 \\
      --adapter /vol/checkpoints/ms_en_lora_bs_4_acc_4/final_adapter

Recipe reference: https://github.com/VinRobotics/model-quantization-recipes/tree/main/recipes/qwen3-asr/vllm
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import modal

load_dotenv(Path(__file__).resolve().parent / ".env")
MODAL_PROFILE = os.environ.get("MODAL_PROFILE", "").strip()

APP_NAME = "qwen3-asr-ms-en-quantize"
VOLUME_NAME = "qwen3-asr-ms-en"
VOL_ROOT = "/vol"
DATA_ROOT = f"{VOL_ROOT}/data"
CKPT_ROOT = f"{VOL_ROOT}/checkpoints"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
GPU = "L4"

_REPO = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.51.0",
        "accelerate",
        "peft>=0.14.0",
        "qwen-asr",
        "nvidia-modelopt",
        "safetensors",
        "soundfile",
        "numpy",
        "tqdm",
        "python-dotenv",
        "huggingface_hub",
    )
    .add_local_python_source("ms_en_quant")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _resolve_adapter_path(adapter: str) -> str:
    adapter = (adapter or "").strip()
    if not adapter:
        raise FileNotFoundError("adapter path is empty")
    candidates = [
        adapter,
        os.path.join(CKPT_ROOT, adapter.lstrip("/")),
        os.path.join(VOL_ROOT, adapter.lstrip("/")),
    ]
    resolved = next((c for c in candidates if os.path.isdir(c)), None)
    if resolved is None:
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    return resolved


@app.function(
    gpu=GPU,
    timeout=60 * 60 * 4,
    volumes={VOL_ROOT: vol},
)
def run_quantize_pipeline(
    action: str = "all",
    fmt: str = "fp8",
    algorithm: str = "max",
    calib_samples: int = 512,
    seed: int = 42,
    adapter: str = "",
    model_id: str = MODEL_ID,
) -> dict:
    import json

    from ms_en_quant.pipeline import default_paths, run_export, run_quantize

    os.environ.setdefault("HF_HOME", f"{CKPT_ROOT}/hf_cache")
    os.makedirs(CKPT_ROOT, exist_ok=True)

    train_jsonl = f"{DATA_ROOT}/train.jsonl"
    if not os.path.isfile(train_jsonl):
        raise FileNotFoundError(
            f"Missing {train_jsonl}. Run:\n"
            f"  modal run train_modal.py --action upload --local-dir ./data_ms_en"
        )

    fmt = fmt.strip().lower()
    if fmt not in ("fp8", "nvfp4"):
        raise ValueError(f"format must be fp8|nvfp4, got {fmt!r}")

    action = action.strip().lower()
    if action not in ("quantize", "export", "all"):
        raise ValueError(f"action must be quantize|export|all, got {action!r}")

    output_pt, tmp_dir, export_dir = default_paths(CKPT_ROOT, fmt)
    adapter_path = _resolve_adapter_path(adapter) if adapter.strip() else None

    result = {
        "format": fmt,
        "algorithm": algorithm,
        "calib_samples": calib_samples,
        "train_jsonl": train_jsonl,
        "output_pt": output_pt,
        "export_dir": export_dir,
        "adapter": adapter_path,
        "model_id": model_id,
    }

    if action in ("quantize", "all"):
        run_quantize(
            model_path=model_id,
            train_jsonl=train_jsonl,
            data_root=DATA_ROOT,
            output_pt=output_pt,
            fmt=fmt,
            algorithm=algorithm,
            calib_samples=calib_samples,
            seed=seed,
            device="cuda:0",
            adapter_path=adapter_path,
            log_dir=os.path.join(CKPT_ROOT, f"qwen3asr_ms_en_{fmt}"),
        )
        vol.commit()

    if action in ("export", "all"):
        if not os.path.isfile(output_pt):
            raise FileNotFoundError(
                f"Missing {output_pt}; run quantize first or pass action=all"
            )
        run_export(
            model_path=model_id,
            quant_pt=output_pt,
            tmp_dir=tmp_dir,
            export_dir=export_dir,
            device="cuda:0",
        )
        vol.commit()

    summary_path = os.path.join(CKPT_ROOT, f"quantize_ms_en_{fmt}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    result["summary_path"] = summary_path
    print(f"[done] {json.dumps(result, indent=2)}")
    return result


@app.local_entrypoint()
def main(
    action: str = "all",
    format: str = "fp8",
    algorithm: str = "max",
    calib_samples: int = 512,
    seed: int = 42,
    adapter: str = "",
    model_id: str = MODEL_ID,
):
    """Run MS-EN ModelOpt quantization on Modal (shared volume with train/loadtest)."""
    import json
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
            f"expected '{expected}' (MODAL_PROFILE)."
        )

    result = run_quantize_pipeline.remote(
        action=action,
        fmt=format,
        algorithm=algorithm,
        calib_samples=calib_samples,
        seed=seed,
        adapter=adapter,
        model_id=model_id or MODEL_ID,
    )
    print("\nResult:")
    print(json.dumps(result, indent=2))
    fmt = format.strip().lower()
    print(
        f"\nLoad-test:\n"
        f"  modal run loadtest_modal.py --backend vllm --quantization {fmt}"
    )
