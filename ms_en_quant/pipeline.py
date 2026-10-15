#!/usr/bin/env python3
"""ModelOpt PTQ for Qwen3-ASR using Malaysian English calibration data.

Adapted from VinRobotics model-quantization-recipes (qwen3-asr/vllm), with
FLEURS/LibriSpeech replaced by local/volume ``data_ms_en/train.jsonl``.

Reference: https://github.com/VinRobotics/model-quantization-recipes/tree/main/recipes/qwen3-asr/vllm
"""

from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import os
import random
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_hf_checkpoint
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"


def resolve_audio_path(path: str, data_root: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    return os.path.join(data_root, path)


def load_ms_en_calib(
    train_jsonl: str,
    data_root: str,
    n_samples: int,
    seed: int = 42,
) -> list[dict]:
    """Load up to ``n_samples`` utterances from train.jsonl for calibration."""
    import soundfile as sf

    rows: list[dict] = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_samples]

    out: list[dict] = []
    for ex in rows:
        ap = resolve_audio_path(ex["audio"], data_root)
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"Missing calibration audio: {ap}")
        wav, sr = sf.read(ap, dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=-1)
        out.append(
            {
                "array": wav,
                "sampling_rate": int(sr),
                "text": ex["text"],
            }
        )
    return out


def _append_quant_exclusions(cfg: dict, patterns: tuple[str, ...]) -> None:
    """Disable module quantizers (ModelOpt list or legacy dict ``quant_cfg``)."""
    quant_cfg = cfg["quant_cfg"]
    if isinstance(quant_cfg, dict):
        for pat in patterns:
            quant_cfg[pat] = {"enable": False}
    elif isinstance(quant_cfg, list):
        for pat in patterns:
            quant_cfg.append({"quantizer_name": pat, "enable": False})
    else:
        raise TypeError(f"unexpected quant_cfg type: {type(quant_cfg)!r}")


def build_quant_cfg(fmt: str, algorithm: str) -> dict:
    if fmt == "fp8":
        cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        if algorithm == "awq_full":
            cfg["algorithm"] = {"method": "awq_full", "alpha_step": 0.1}
        elif algorithm == "mse":
            cfg["algorithm"] = "mse"
    elif fmt == "nvfp4":
        if algorithm == "awq_full":
            cfg = copy.deepcopy(mtq.NVFP4_AWQ_FULL_CFG)
        else:
            cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
            if algorithm == "mse":
                cfg["algorithm"] = {
                    "method": "mse",
                    "num_steps": 20,
                    "start_multiplier": 0.2,
                    "stop_multiplier": 4.5,
                }
    else:
        raise ValueError(f"Unknown format {fmt!r}; use fp8|nvfp4")

    _append_quant_exclusions(cfg, ("*audio_tower*", "*lm_head*"))
    return cfg


def resolve_model_snapshot_dir(model_path: str) -> str:
    """Local directory for ``copytree`` (HF hub id → cached snapshot)."""
    if os.path.isdir(model_path):
        return os.path.abspath(model_path)
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=model_path)
    print(f"[model] resolved {model_path} → {local}")
    return local


def _load_model(
    model_path: str,
    device: str,
    adapter_path: str | None = None,
):
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
        Qwen3ASRProcessor,
    )

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    )
    processor = Qwen3ASRProcessor.from_pretrained(
        model_path,
        fix_mistral_regex=True,
    )

    if adapter_path:
        from peft import PeftModel

        print(f"[quantize] loading LoRA adapter: {adapter_path}")
        model.thinker = PeftModel.from_pretrained(model.thinker, adapter_path)
        model.thinker = model.thinker.merge_and_unload()
        print("[quantize] merged adapter into base thinker")

    return model, processor


def run_quantize(
    *,
    model_path: str,
    train_jsonl: str,
    data_root: str,
    output_pt: str,
    fmt: str,
    algorithm: str = "max",
    calib_samples: int = 512,
    seed: int = 42,
    device: str = "cuda:0",
    adapter_path: str | None = None,
    log_dir: str | None = None,
) -> str:
    model, processor = _load_model(model_path, device, adapter_path)
    dev = next(model.parameters()).device

    calib = load_ms_en_calib(train_jsonl, data_root, calib_samples, seed=seed)
    print(f"[quantize] calibration samples: {len(calib)} from {train_jsonl}")
    print(f"[quantize] format={fmt} algorithm={algorithm}")

    def forward_step(thinker, data):
        inputs = processor(
            audio=data["array"],
            sampling_rate=data["sampling_rate"],
            text=data["text"],
            return_tensors="pt",
        )
        inputs = {
            k: v.to(dtype=torch.bfloat16, device=dev)
            if v.dtype == torch.float32
            else v.to(device=dev)
            for k, v in inputs.items()
        }
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return thinker(
            input_ids=inputs["input_ids"],
            input_features=inputs.get("input_features"),
            attention_mask=inputs.get("attention_mask"),
            feature_attention_mask=inputs.get("feature_attention_mask"),
            labels=labels,
        )

    cfg = build_quant_cfg(fmt, algorithm)
    thinker = model.thinker

    def forward_loop(m):
        for data in calib:
            forward_step(m, data)

    thinker = mtq.quantize(thinker, cfg, forward_loop)

    log_base = log_dir or os.path.dirname(output_pt) or "."
    os.makedirs(log_base, exist_ok=True)
    log_path = os.path.join(log_base, f"quantize_decisions_{fmt}.txt")
    buf = io.StringIO()
    with redirect_stdout(buf):
        mtq.print_quant_summary(thinker)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"[quantize] decisions log → {log_path}")

    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    mto.save(thinker, output_pt)
    print(f"[quantize] saved → {output_pt}")

    del model
    torch.cuda.empty_cache()
    return output_pt


def run_export(
    *,
    model_path: str,
    quant_pt: str,
    tmp_dir: str,
    export_dir: str,
    device: str = "cuda:0",
) -> str:
    model, _ = _load_model(model_path, device, adapter_path=None)

    print("[export] ModelOpt HF export …")
    mto.restore(model.thinker, quant_pt)
    with torch.inference_mode():
        export_hf_checkpoint(model.thinker, export_dir=tmp_dir)
    del model
    torch.cuda.empty_cache()
    print(f"[export] tmp HF quant weights → {tmp_dir}")

    print("[export] building vLLM checkpoint …")
    base_dir = resolve_model_snapshot_dir(model_path)
    shutil.copytree(base_dir, export_dir, dirs_exist_ok=True)
    shutil.copy(
        f"{tmp_dir}/hf_quant_config.json",
        f"{export_dir}/hf_quant_config.json",
    )
    for f in glob.glob(f"{export_dir}/model-*.safetensors"):
        os.remove(f)
    index_path = f"{export_dir}/model.safetensors.index.json"
    if os.path.exists(index_path):
        os.remove(index_path)

    tensors = {}
    for shard in sorted(glob.glob(f"{tmp_dir}/*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                tensors[f"thinker.{key}"] = f.get_tensor(key)

    os.makedirs(export_dir, exist_ok=True)
    save_file(tensors, f"{export_dir}/model.safetensors")
    print(f"[export] vLLM checkpoint → {export_dir}")
    return export_dir


def default_paths(ckpt_root: str, fmt: str) -> tuple[str, str, str]:
    """Return (quant_pt, tmp_dir, export_dir) under ckpt_root."""
    if fmt == "fp8":
        tag = "fp8"
    elif fmt == "nvfp4":
        tag = "nvfp4"
    else:
        raise ValueError(fmt)
    base = os.path.join(ckpt_root, f"qwen3asr_ms_en_{tag}")
    return (
        f"{base}.pt",
        f"{base}_hf_tmp",
        f"{base}_vllm",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MS-EN ModelOpt quantize + vLLM export")
    p.add_argument(
        "action",
        choices=("quantize", "export", "all"),
        help="quantize: .pt only; export: vLLM dir; all: both",
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--train-jsonl",
        default="./data_ms_en/train.jsonl",
        help="Malaysian English train JSONL (prep_data.py output)",
    )
    p.add_argument(
        "--data-root",
        default=".",
        help="Root for resolving relative audio paths in JSONL",
    )
    p.add_argument("--format", required=True, choices=["fp8", "nvfp4"])
    p.add_argument(
        "--algorithm",
        default="max",
        choices=["max", "mse", "awq_full"],
    )
    p.add_argument("--calib-samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter", default="", help="Optional PEFT adapter to merge before PTQ")
    p.add_argument("--output-pt", default="")
    p.add_argument("--tmp-dir", default="")
    p.add_argument("--export-dir", default="")
    p.add_argument("--ckpt-root", default="./quant_out")
    args = p.parse_args(argv)

    fmt = args.format
    pt, tmp, exp = default_paths(args.ckpt_root, fmt)
    output_pt = args.output_pt or pt
    tmp_dir = args.tmp_dir or tmp
    export_dir = args.export_dir or exp
    adapter = (args.adapter or "").strip() or None

    data_root = os.path.abspath(args.data_root)
    train_jsonl = os.path.abspath(args.train_jsonl)
    if args.action in ("quantize", "all") and not os.path.isfile(train_jsonl):
        raise FileNotFoundError(train_jsonl)

    if args.action in ("quantize", "all"):
        run_quantize(
            model_path=args.model_path,
            train_jsonl=train_jsonl,
            data_root=data_root,
            output_pt=output_pt,
            fmt=fmt,
            algorithm=args.algorithm,
            calib_samples=args.calib_samples,
            seed=args.seed,
            device=args.device,
            adapter_path=adapter,
        )

    if args.action in ("export", "all"):
        if not os.path.isfile(output_pt):
            raise FileNotFoundError(f"Missing quant checkpoint: {output_pt}")
        run_export(
            model_path=args.model_path,
            quant_pt=output_pt,
            tmp_dir=tmp_dir,
            export_dir=export_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
