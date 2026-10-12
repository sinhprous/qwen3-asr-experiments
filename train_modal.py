#!/usr/bin/env python3
"""Fine-tune Qwen3-ASR-1.7B (LoRA) on Malaysian English via Modal.

Uses Modal profile from ``MODAL_PROFILE`` (see ``.env``), L4 GPU (~$0.80/hr),
and the official Qwen3-ASR SFT protocol with left-pad-aware label masking.

Workflow
--------
1) Prep locally (once)::

    pip install -r requirements.txt
    python prep_data.py \\
      --src "/Users/sinh/Downloads/1784486741842-cv26-malaysian-english" \\
      --out ./data_ms_en

2) Select Modal profile, then upload data::

    export MODAL_PROFILE=...   # or: modal profile activate <your-profile>
    modal run train_modal.py --action upload --local-dir ./data_ms_en

3) Train + evaluate (baseline WER → LoRA train → post WER)::

    modal run train_modal.py --action train

4) Eval-only on test (base model, or a saved adapter)::

    modal run train_modal.py --action eval
    modal run train_modal.py --action eval \\
      --adapter /vol/checkpoints/ms_en_lora/final_adapter

5) Re-eval base + adapter with per-utterance WER compare::

    modal run train_modal.py --action eval_compare \\
      --adapter /vol/checkpoints/ms_en_lora/final_adapter \\
      --output-name ms_en_lora
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import modal

# Local-only; Modal remote containers do not need .env.
load_dotenv(Path(__file__).resolve().parent / ".env")
MODAL_PROFILE = os.environ.get("MODAL_PROFILE", "").strip()

APP_NAME = "qwen3-asr-ms-en"
VOLUME_NAME = "qwen3-asr-ms-en"
VOL_ROOT = "/vol"
DATA_ROOT = f"{VOL_ROOT}/data"
CKPT_ROOT = f"{VOL_ROOT}/checkpoints"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# Cost-effective default: L4 24GB. Switch to "A10" if OOM.
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


# ── Helpers used inside Modal containers ─────────────────────────────────────


def _resolve_audio_path(path: str, data_root: str) -> str:
    import os

    if os.path.isabs(path) and os.path.exists(path):
        return path
    return os.path.join(data_root, path)


def _rewrite_jsonl_paths(src_jsonl: str, dst_jsonl: str, data_root: str) -> int:
    """Copy JSONL with audio paths resolved under data_root."""
    import json
    import os

    n = 0
    parent = os.path.dirname(dst_jsonl)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(src_jsonl, encoding="utf-8") as fin, open(
        dst_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ex["audio"] = _resolve_audio_path(ex["audio"], data_root)
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n


def _patch_outer_forward(model) -> None:
    """Route HF Trainer calls into model.thinker.forward (official SFT script)."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def _make_preprocess_fn(processor):
    def _fn(ex):
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        prefix_text = processor.apply_chat_template(
            [msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _fn


def _build_collator(processor, sampling_rate: int = 16000, max_audio_seconds: float = 30.0):
    from dataclasses import dataclass
    from typing import Any, Dict, List

    import librosa
    import torch

    @dataclass
    class Collator:
        processor: Any
        sampling_rate: int = 16000
        max_audio_samples: int = int(max_audio_seconds * 16000)

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            eos = self.processor.tokenizer.eos_token or ""
            prefix_texts = [f["prefix_text"] for f in features]
            targets = [f["target"] for f in features]
            full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

            audios = []
            for f in features:
                wav, _ = librosa.load(f["audio"], sr=self.sampling_rate, mono=True)
                audios.append(wav[: self.max_audio_samples])

            full_inputs = self.processor(
                text=full_texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            prefix_inputs = self.processor(
                text=prefix_texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            # Left-pad-aware masking (fixes official script bug #70 / #216)
            labels = full_inputs["input_ids"].clone()
            seq_len = labels.shape[1]
            full_lens = full_inputs["attention_mask"].sum(dim=1).tolist()
            prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
            for i, (fl, pl) in enumerate(zip(full_lens, prefix_lens)):
                pad_offset = seq_len - int(fl)
                labels[i, : pad_offset + int(pl)] = -100

            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            full_inputs["labels"] = labels
            return full_inputs

    return Collator(processor=processor, sampling_rate=sampling_rate)


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


def _utt_wer_stats(ref: str, hyp: str) -> dict:
    """Per-utt WER with punctuation kept and with punctuation stripped."""
    from jiwer import process_words

    ref_n = _normalize_for_wer(ref, strip_punct=False)
    hyp_n = _normalize_for_wer(hyp, strip_punct=False)
    ref_np = _normalize_for_wer(ref, strip_punct=True)
    hyp_np = _normalize_for_wer(hyp, strip_punct=True)

    def _one(ref_t: str, hyp_t: str) -> dict:
        if not ref_t.strip():
            return {
                "wer": 0.0 if not hyp_t.strip() else 1.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
                "ref_words": 0,
            }
        out = process_words(ref_t, hyp_t)
        ref_words = max(1, int(out.hits + out.substitutions + out.deletions))
        return {
            "wer": float(out.wer),
            "substitutions": int(out.substitutions),
            "deletions": int(out.deletions),
            "insertions": int(out.insertions),
            "hits": int(out.hits),
            "ref_words": ref_words,
        }

    with_p = _one(ref_n, hyp_n)
    no_p = _one(ref_np, hyp_np)
    return {
        "ref": ref_n,
        "hyp": hyp_n,
        "ref_no_punct": ref_np,
        "hyp_no_punct": hyp_np,
        "wer": with_p["wer"],
        "wer_no_punct": no_p["wer"],
        "substitutions": with_p["substitutions"],
        "deletions": with_p["deletions"],
        "insertions": with_p["insertions"],
        "hits": with_p["hits"],
        "ref_words": with_p["ref_words"],
        "substitutions_no_punct": no_p["substitutions"],
        "deletions_no_punct": no_p["deletions"],
        "insertions_no_punct": no_p["insertions"],
        "hits_no_punct": no_p["hits"],
        "ref_words_no_punct": no_p["ref_words"],
    }


def _corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    from jiwer import process_words

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


def _compute_wer(
    references: list[str],
    hypotheses: list[str],
    audios: list[str] | None = None,
    languages: list[str] | None = None,
) -> dict:
    """Corpus WER with punctuation kept (`wer`) and stripped (`wer_no_punct`)."""
    audios = audios or [""] * len(references)
    languages = languages or [""] * len(references)

    per_utt = []
    kept_refs, kept_hyps = [], []
    kept_refs_np, kept_hyps_np = [], []
    for i, (ref_raw, hyp_raw, audio, lang) in enumerate(
        zip(references, hypotheses, audios, languages)
    ):
        stats = _utt_wer_stats(ref_raw, hyp_raw)
        stats["audio"] = audio
        stats["language"] = lang or ""
        stats["index"] = i
        per_utt.append(stats)
        if stats["ref"].strip():
            kept_refs.append(stats["ref"])
            kept_hyps.append(stats["hyp"])
            kept_refs_np.append(stats["ref_no_punct"])
            kept_hyps_np.append(stats["hyp_no_punct"])

    with_p = _corpus_wer(kept_refs, kept_hyps)
    no_p = _corpus_wer(kept_refs_np, kept_hyps_np)
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
        "per_utt": per_utt,
        "wer_protocol": {
            "wer": "strip_<asr_text>, lowercase, keep punctuation, collapse whitespace",
            "wer_no_punct": (
                "strip_<asr_text>, lowercase, remove string.punctuation, "
                "collapse whitespace"
            ),
        },
    }


def _build_per_utt_compare(before: dict, after: dict) -> list[dict]:
    """Join before/after per-utt rows on audio path (fallback: index)."""
    after_by_audio = {u["audio"]: u for u in after.get("per_utt", []) if u.get("audio")}
    after_by_idx = {u["index"]: u for u in after.get("per_utt", [])}
    rows = []
    for b in before.get("per_utt", []):
        a = after_by_audio.get(b.get("audio")) or after_by_idx.get(b.get("index"))
        if a is None:
            continue
        wb = float(b.get("wer", 0.0) or 0.0)
        wa = float(a.get("wer", 0.0) or 0.0)
        wb_np = float(b.get("wer_no_punct", 0.0) or 0.0)
        wa_np = float(a.get("wer_no_punct", 0.0) or 0.0)
        rows.append(
            {
                "index": b.get("index"),
                "audio": b.get("audio"),
                "ref": b.get("ref"),
                "ref_no_punct": b.get("ref_no_punct"),
                "language_before": b.get("language"),
                "language_after": a.get("language"),
                "hyp_before": b.get("hyp"),
                "hyp_after": a.get("hyp"),
                "hyp_before_no_punct": b.get("hyp_no_punct"),
                "hyp_after_no_punct": a.get("hyp_no_punct"),
                "wer_before": wb,
                "wer_after": wa,
                "wer_delta": wa - wb,
                "wer_no_punct_before": wb_np,
                "wer_no_punct_after": wa_np,
                "wer_no_punct_delta": wa_np - wb_np,
                "ref_words": b.get("ref_words"),
                "ref_words_no_punct": b.get("ref_words_no_punct"),
            }
        )
    rows.sort(key=lambda r: (r["wer_delta"], -float(r["wer_before"] or 0.0)))
    return rows


def _load_jsonl(path: str) -> list[dict]:
    import json

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _transcribe_jsonl(
    model,
    jsonl_path: str,
    data_root: str,
    batch_size: int = 4,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (refs, hyps, languages, audio_ids) for every row in jsonl."""
    import os

    from tqdm import tqdm

    rows = _load_jsonl(jsonl_path)
    refs, hyps, langs, audios = [], [], [], []

    for start in tqdm(range(0, len(rows), batch_size), desc="transcribe"):
        batch = rows[start : start + batch_size]
        paths = [_resolve_audio_path(r["audio"], data_root) for r in batch]
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        results = model.transcribe(audio=paths)
        for row, res in zip(batch, results):
            refs.append(row["text"])
            hyps.append(getattr(res, "text", "") or "")
            langs.append(getattr(res, "language", "") or "")
            # Keep the JSONL-relative id for stable before/after joins
            audios.append(row["audio"])
    return refs, hyps, langs, audios


def _evaluate(
    model_path: str,
    adapter_path: str | None,
    test_jsonl: str,
    data_root: str,
    dtype,
    device: str = "cuda:0",
) -> dict:
    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    print(f"[eval] base={model_path} adapter={adapter_path or '(none)'}")
    wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )

    if adapter_path:
        thinker = wrapper.model.thinker
        wrapper.model.thinker = PeftModel.from_pretrained(thinker, adapter_path)
        wrapper.model.thinker.eval()
        print(f"[eval] loaded adapter from {adapter_path}")

    refs, hyps, langs, audios = _transcribe_jsonl(wrapper, test_jsonl, data_root)
    metrics = _compute_wer(refs, hyps, audios=audios, languages=langs)

    # Keep a short preview for console/summary; full detail is in per_utt
    metrics["examples"] = [
        {
            "audio": u["audio"],
            "ref": u["ref"],
            "hyp": u["hyp"],
            "language": u["language"],
            "wer": u["wer"],
            "wer_no_punct": u.get("wer_no_punct"),
        }
        for u in metrics.get("per_utt", [])[:10]
    ]
    print(
        f"[eval] WER={metrics['wer'] * 100:.2f}%  "
        f"WER(no punct)={metrics['wer_no_punct'] * 100:.2f}%  "
        f"n={metrics['n']}"
    )

    del wrapper
    torch.cuda.empty_cache()
    return metrics


def _run_training(
    train_jsonl: str,
    eval_jsonl: str | None,
    output_dir: str,
    model_path: str,
    dtype,
) -> str:
    import os

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from qwen_asr import Qwen3ASRModel
    from transformers import (
        EarlyStoppingCallback,
        GenerationConfig,
        Trainer,
        TrainingArguments,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Cap epochs high; EarlyStoppingCallback exits earlier on flat eval_loss.
    MAX_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 4  # within 3–5; stop after this many worse evals
    EVAL_STEPS = 50

    wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=None,
        attn_implementation="sdpa",
    )
    model = wrapper.model.cuda()
    processor = wrapper.processor

    _patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    # Avoid HF Trainer loss scaling weirdness with patched forward (**kwargs)
    model.accepts_loss_kwargs = False

    model.requires_grad_(False)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()

    data_files = {"train": train_jsonl}
    if eval_jsonl and os.path.exists(eval_jsonl):
        data_files["validation"] = eval_jsonl

    ds = load_dataset("json", data_files=data_files)
    ds = ds.map(_make_preprocess_fn(processor), num_proc=1)
    keep = {"audio", "target", "prefix_text"}
    for split in ds:
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = _build_collator(processor)

    has_val = "validation" in ds
    if not has_val:
        print(
            "[train] WARNING: no validation set — early stopping disabled; "
            f"will run all {MAX_EPOCHS} epochs."
        )

    use_bf16 = dtype == torch.bfloat16
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # 8,  # effective batch 16
        learning_rate=2e-5,
        num_train_epochs=MAX_EPOCHS,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        # NOTE: load_best_model_at_end=False on purpose.
        # Trainer._load_best_model() uses torch.load on optimizer/rng .pt files, which
        # fails on torch<2.6 with current transformers (CVE-2025-32434). Early stopping
        # still tracks best_model_checkpoint; we copy that dir to final_adapter below.
        save_strategy="steps" if has_val else "epoch",
        save_steps=EVAL_STEPS if has_val else None,
        save_total_limit=3,
        save_safetensors=True,
        eval_strategy="steps" if has_val else "no",
        eval_steps=EVAL_STEPS if has_val else None,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        gradient_checkpointing=False,
    )

    class CastFloatTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            mdtype = next(self.model.parameters()).dtype
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=mdtype)
            return inputs

        def _save(self, output_dir=None, state_dict=None):
            output_dir = output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            thinker = self.model.thinker
            thinker.save_pretrained(output_dir)
            self.processing_class.save_pretrained(output_dir)

    callbacks = []
    if has_val:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=0.0,
            )
        )
        print(
            f"[train] early stopping on eval_loss: "
            f"patience={EARLY_STOPPING_PATIENCE}, max_epochs={MAX_EPOCHS}, "
            f"eval_steps={EVAL_STEPS}"
        )

    trainer = CastFloatTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
        processing_class=processor.tokenizer,
        callbacks=callbacks,
    )

    train_out = trainer.train()
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    print(
        f"[train] finished: global_step={train_out.global_step} "
        f"best_metric={best_metric} best_ckpt={best_ckpt}"
    )

    final_dir = os.path.join(output_dir, "final_adapter")
    if best_ckpt and os.path.isdir(best_ckpt):
        # Prefer the early-stopped best adapter dir (custom _save writes PEFT files there).
        import shutil

        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(best_ckpt, final_dir)
        print(f"[train] copied best checkpoint → {final_dir}")
    else:
        os.makedirs(final_dir, exist_ok=True)
        model.thinker.save_pretrained(final_dir)
        processor.tokenizer.save_pretrained(final_dir)
        print(f"[train] adapter saved → {final_dir}")
    return final_dir


def _compact(result: dict) -> dict:
    """Drop bulky per-utt / example transcripts for console print."""
    drop = {"examples", "per_utt"}
    out = {}
    for k, v in result.items():
        if isinstance(v, dict) and (drop & set(v.keys())):
            out[k] = {kk: vv for kk, vv in v.items() if kk not in drop}
        elif k == "per_utt_compare" and isinstance(v, list):
            out[k] = f"<{len(v)} utterances — see wer_per_utt_compare.json>"
        else:
            out[k] = v
    return out


def _resolve_adapter_path(adapter: str) -> str:
    import os

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


def _write_json(path: str, obj) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[write] {path}")


# ── Modal remote function ────────────────────────────────────────────────────


@app.function(
    volumes={VOL_ROOT: vol},
    gpu=GPU,
    timeout=60 * 60 * 4,
    memory=32768,
)
def run_pipeline(
    eval_only: bool = False,
    eval_compare: bool = False,
    adapter: str = "",
    skip_baseline: bool = False,
    output_name: str = "ms_en_lora",
) -> dict:
    """Train (optional) and/or evaluate on the test set."""
    import gc
    import os

    import torch

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", f"{CKPT_ROOT}/hf_cache")
    os.makedirs(CKPT_ROOT, exist_ok=True)

    needs_train_data = not (eval_only or eval_compare)
    train_src = f"{DATA_ROOT}/train.jsonl"
    dev_src = f"{DATA_ROOT}/dev.jsonl"
    test_src = f"{DATA_ROOT}/test.jsonl"
    for p in (train_src if needs_train_data else test_src, test_src):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Upload prepared data first:\n"
                f"  export MODAL_PROFILE=<your-profile>\n"
                f"  modal run train_modal.py --action upload --local-dir ./data_ms_en"
            )

    work = f"{CKPT_ROOT}/work"
    os.makedirs(work, exist_ok=True)
    train_jsonl = f"{work}/train.jsonl"
    dev_jsonl = f"{work}/dev.jsonl"
    test_jsonl = f"{work}/test.jsonl"

    if needs_train_data:
        _rewrite_jsonl_paths(train_src, train_jsonl, DATA_ROOT)
        if os.path.exists(dev_src):
            _rewrite_jsonl_paths(dev_src, dev_jsonl, DATA_ROOT)
        else:
            dev_jsonl = None
    else:
        train_jsonl = ""
        dev_jsonl = None
    _rewrite_jsonl_paths(test_src, test_jsonl, DATA_ROOT)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(
        f"[info] gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
        f"dtype={dtype}"
    )

    out_dir = f"{CKPT_ROOT}/{output_name}"
    os.makedirs(out_dir, exist_ok=True)
    results: dict = {
        "eval_only": eval_only,
        "eval_compare": eval_compare,
        "gpu": GPU,
        "model": MODEL_ID,
    }

    adapter_path = _resolve_adapter_path(adapter) if adapter.strip() else None

    def _save_compare(before: dict, after: dict) -> list[dict]:
        compare = _build_per_utt_compare(before, after)
        _write_json(os.path.join(out_dir, "wer_per_utt_compare.json"), compare)
        improved = sum(1 for r in compare if r["wer_delta"] < 0)
        worsened = sum(1 for r in compare if r["wer_delta"] > 0)
        unchanged = sum(1 for r in compare if r["wer_delta"] == 0)
        improved_np = sum(1 for r in compare if r["wer_no_punct_delta"] < 0)
        worsened_np = sum(1 for r in compare if r["wer_no_punct_delta"] > 0)
        unchanged_np = sum(1 for r in compare if r["wer_no_punct_delta"] == 0)
        print(
            f"[compare] punct: n={len(compare)} improved={improved} "
            f"worsened={worsened} unchanged={unchanged}"
        )
        print(
            f"[compare] no punct: n={len(compare)} improved={improved_np} "
            f"worsened={worsened_np} unchanged={unchanged_np}"
        )
        return compare

    # ── Eval both base + adapter (no training) ──
    if eval_compare:
        if adapter_path is None:
            adapter_path = _resolve_adapter_path(
                f"{CKPT_ROOT}/{output_name}/final_adapter"
            )
        results["adapter"] = adapter_path

        before = _evaluate(
            model_path=MODEL_ID,
            adapter_path=None,
            test_jsonl=test_jsonl,
            data_root=DATA_ROOT,
            dtype=dtype,
        )
        results["wer_before"] = before
        _write_json(os.path.join(out_dir, "wer_before.json"), before)
        vol.commit()

        gc.collect()
        torch.cuda.empty_cache()

        after = _evaluate(
            model_path=MODEL_ID,
            adapter_path=adapter_path,
            test_jsonl=test_jsonl,
            data_root=DATA_ROOT,
            dtype=dtype,
        )
        results["wer_after"] = after
        _write_json(os.path.join(out_dir, "wer_after.json"), after)

        results["wer_delta"] = after["wer"] - before["wer"]
        results["wer_no_punct_delta"] = after["wer_no_punct"] - before["wer_no_punct"]
        results["per_utt_compare"] = _save_compare(before, after)
        print(
            f"\n=== WER summary (test) ===\n"
            f"  before: {before['wer'] * 100:.2f}%  "
            f"(no punct {before['wer_no_punct'] * 100:.2f}%)\n"
            f"  after:  {after['wer'] * 100:.2f}%  "
            f"(no punct {after['wer_no_punct'] * 100:.2f}%)\n"
            f"  delta:  {results['wer_delta'] * 100:+.2f}%  "
            f"(no punct {results['wer_no_punct_delta'] * 100:+.2f}%)\n"
        )
        _write_json(os.path.join(out_dir, "summary.json"), results)
        vol.commit()
        return results

    if eval_only:
        metrics = _evaluate(
            model_path=MODEL_ID,
            adapter_path=adapter_path,
            test_jsonl=test_jsonl,
            data_root=DATA_ROOT,
            dtype=dtype,
        )
        tag = "after" if adapter_path else "before"
        # Keep legacy baseline filename for base-only eval
        filename = "wer_after.json" if adapter_path else "wer_before.json"
        results[f"wer_{tag}"] = metrics
        if not adapter_path:
            results["wer_baseline"] = metrics
        _write_json(os.path.join(out_dir, filename), metrics)
        vol.commit()
        return results

    # Full pipeline: baseline → train → after
    if not skip_baseline:
        baseline = _evaluate(
            model_path=MODEL_ID,
            adapter_path=None,
            test_jsonl=test_jsonl,
            data_root=DATA_ROOT,
            dtype=dtype,
        )
        results["wer_before"] = baseline
        _write_json(os.path.join(out_dir, "wer_before.json"), baseline)
        vol.commit()

    gc.collect()
    torch.cuda.empty_cache()

    final_adapter = _run_training(
        train_jsonl=train_jsonl,
        eval_jsonl=dev_jsonl,
        output_dir=out_dir,
        model_path=MODEL_ID,
        dtype=dtype,
    )
    results["adapter"] = final_adapter
    vol.commit()

    gc.collect()
    torch.cuda.empty_cache()

    after = _evaluate(
        model_path=MODEL_ID,
        adapter_path=final_adapter,
        test_jsonl=test_jsonl,
        data_root=DATA_ROOT,
        dtype=dtype,
    )
    results["wer_after"] = after
    _write_json(os.path.join(out_dir, "wer_after.json"), after)

    if "wer_before" in results:
        before = results["wer_before"]
        before_wer = before["wer"]
        after_wer = after["wer"]
        before_np = before["wer_no_punct"]
        after_np = after["wer_no_punct"]
        results["wer_delta"] = after_wer - before_wer
        results["wer_no_punct_delta"] = after_np - before_np
        results["per_utt_compare"] = _save_compare(before, after)
        print(
            f"\n=== WER summary (test) ===\n"
            f"  before: {before_wer * 100:.2f}%  (no punct {before_np * 100:.2f}%)\n"
            f"  after:  {after_wer * 100:.2f}%  (no punct {after_np * 100:.2f}%)\n"
            f"  delta:  {(after_wer - before_wer) * 100:+.2f}%  "
            f"(no punct {(after_np - before_np) * 100:+.2f}%)\n"
        )

    _write_json(os.path.join(out_dir, "summary.json"), results)
    vol.commit()
    print(f"[done] summary → {os.path.join(out_dir, 'summary.json')}")
    return results


# ── Local entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    action: str = "train",
    local_dir: str = "./data_ms_en",
    adapter: str = "",
    skip_baseline: bool = False,
    output_name: str = "ms_en_lora",
):
    """Entry point.

    Actions:
      - upload       : push ./data_ms_en into the Modal volume at /vol/data
      - train        : baseline WER → LoRA fine-tune → post WER (default)
      - eval         : test-set WER only (base model, or --adapter …)
      - eval_compare : base + adapter on test, write per-utt compare JSON
    """
    import json
    import os
    import subprocess
    import sys

    action = action.lower().strip()
    if action not in {"upload", "train", "eval", "eval_compare"}:
        raise SystemExit(
            f"Unknown action={action!r}. Use upload|train|eval|eval_compare"
        )

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

    if action == "upload":
        local_dir = os.path.abspath(local_dir)
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(local_dir)
        required = ["train.jsonl", "dev.jsonl", "test.jsonl", "wavs"]
        missing = [
            r for r in required if not os.path.exists(os.path.join(local_dir, r))
        ]
        if missing:
            raise FileNotFoundError(
                f"Local data missing {missing}. Run prep_data.py first."
            )

        print(f"Ensuring volume '{VOLUME_NAME}' exists…")
        create = subprocess.run(
            [sys.executable, "-m", "modal", "volume", "create", VOLUME_NAME],
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            err = (create.stderr or create.stdout or "").lower()
            if "already" not in err and "exist" not in err:
                print(create.stdout)
                print(create.stderr)
                sys.exit(create.returncode)

        # Maps local data_ms_en/{train,dev,test}.jsonl + wavs/ → volume:/data/...
        cmd = [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "put",
            VOLUME_NAME,
            local_dir,
            "/data",
            "--force",
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            sys.exit(proc.returncode)
        print(f"Upload complete → volume {VOLUME_NAME}:/data")
        return

    if action == "eval_compare" and not adapter.strip():
        adapter = f"/vol/checkpoints/{output_name}/final_adapter"

    result = run_pipeline.remote(
        eval_only=(action == "eval"),
        eval_compare=(action == "eval_compare"),
        adapter=adapter,
        skip_baseline=skip_baseline,
        output_name=output_name,
    )
    print("\nResult:")
    print(json.dumps(_compact(result), indent=2))
