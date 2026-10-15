# Qwen3-ASR fine-tuning and inference optimization

## Project layout

| File | Role |
|------|------|
| `prep_data.py` | Local Common Voice -> JSONL + 16 kHz WAVs |
| `train_modal.py` | Modal upload / train / eval / eval_compare |
| `bench_modal.py` | Modal single-stream RTF and throughput baseline |
| `loadtest_modal.py` | Modal concurrent-stream load test (`--backend pytorch` / `vllm`) |
| `quantize_modal.py` | ModelOpt FP8 / NVFP4 PTQ on MS-EN calibration data -> vLLM export on volume |
| `ms_en_quant/` | Quantization + export implementation (used by `quantize_modal.py`) |
| `test_wer.py` | Isolated WER protocol check (`--self-test` / `--ref` / `--hyp`) |
| `requirements.txt` | Local prep dependencies |
| `.env.example` | `HF_TOKEN=` placeholder |

**NOTE**: We use [Modal](https://modal.com/) serverless infrastructure for running experiment. Why? Because we can run training/inference with sub-second cold starts from local machine on GPU infrastructure. And Modal natively supports Python.

---

## Setup

1. Create a Modal account and follow [Modal getting started](https://modal.com/docs/guide/getting-started).
2. Use **your own** Modal profile or workspace and activate it before any `modal` command:

```bash
modal profile activate <your-profile>
# or: export MODAL_PROFILE=<your-profile>
```

3. Install dependencies:

```bash
pip install -r requirements.txt
pip install modal huggingface_hub
modal profile activate <your-profile>
```

Optional: copy `.env.example` -> `.env` and set `HF_TOKEN=...` for Hugging Face downloads.

### Data

We use the **Common Voice Malaysian English** corpus for fine-tuning and evaluation. Download and extract the dataset from the [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmrt71n6q001fmm07u64adwuv), then convert it (folder with `train/dev/test.tsv` + `clips/`):

```bash
python prep_data.py \
  --src "/path/to/cv26-malaysian-english" \
  --out ./data_ms_en
```

This writes `data_ms_en/{train,dev,test}.jsonl` and `data_ms_en/wavs/` (16 kHz mono). For the published run, split sizes are **2343 / 62 / 71** train / dev / test utterances.

Upload to **your** Modal volume:

```bash
modal run train_modal.py --action upload --local-dir ./data_ms_en
```

---

## Part 1. Fine-tuning

We use **LoRA** rather than full-parameter fine-tuning. For a corpus of only 3.56 hours of speech, LoRA is cheaper to train and reduces the risk of catastrophic forgetting on the base model.

Hyperparameters for finetuning (defaults in `train_modal.py`):

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen3-ASR-1.7B` |
| GPU | Modal **L4** |
| LoRA | r=16, α=32, dropout=0.05, `target_modules="all-linear"` |
| Train batch | `per_device_train_batch_size=4`, `gradient_accumulation_steps=4` (effective **16**) |
| LR | 2e-5, cosine, `warmup_ratio=0.03` |
| Max epochs | 10, early stopping patience=4 on `eval_loss` |
| Precision | bfloat16 on L4 |

```bash
modal run train_modal.py \
  --action train \
  --output-name <experiment_name>
```

Pipeline: baseline test WER -> LoRA training -> post-training test WER. Artifacts on your volume:

```
/vol/checkpoints/<experiment_name>/
  final_adapter/
  wer_before.json
  wer_after.json
  wer_per_utt_compare.json
  summary.json
  checkpoint-*/
```

### Evaluation metrics

Evaluation reports **two** corpus-level WERs (same hypotheses, different text normalization):

| Metric | Normalization before `jiwer.process_words` |
|--------|--------------------------------------------|
| **WER** | strip `<asr_text>` -> lowercase -> **keep punctuation** -> collapse whitespace |
| **WER (no punct)** | same, but also remove `string.punctuation` |

Empty references (after punct-kept normalization) are skipped. Sanity check: `python test_wer.py --self-test`.

### Training results

Test set **n=71** (both metrics use the same saved hypotheses):

| Model | WER | Sub / Del / Ins | Hits | WER (no punct) | Sub / Del / Ins | Hits |
|-------|-----|-----------------|------|----------------|-----------------|------|
| Base `Qwen3-ASR-1.7B` | **13.37%** | 65 / 4 / 2 | 462 | **6.62%** | 31 / 2 / 2 | 496 |
| + LoRA | **8.66%** | 41 / 4 / 1 | 486 | **5.29%** | 25 / 2 / 1 | 502 |
| Δ | **−4.71 abs** | | | **−1.32 abs** | | |

Per-utterance comparison:

- punctuation kept: **22 improved**, **2 worsened**, **47 unchanged**
- no punctuation: **6 improved**, **1 worsened**, **64 unchanged**

You can check utterance-level details at `wer_per_utt_compare.json` file in the training artifact folder and see that the training actually modifies the model and the model indeed generate better result. For example with the audio `test_common_voice_en_31098685.wav`

- Reference (Groundtruth): `it introduced different style of book design and publishing for its time.`
- Baseline hypothesis: `it introduced the first style book design and publishing for its time.`
- Fine-tuned hypothesis: `it introduced different styles of book design and publishing for its time.`

### Reproduce eval (Hugging Face adapter)

The trained adapter weights are uploaded to [`sinhprous/qwen3-asr-1.7b-ms-en-lora`](https://huggingface.co/sinhprous/qwen3-asr-1.7b-ms-en-lora). Reproduce WER on your own Modal account:

1. Prepare and upload data (see Setup).
2. Download the adapter and put it into your volume:

```bash
huggingface-cli download sinhprous/qwen3-asr-1.7b-ms-en-lora \
  --local-dir ./ms_en_lora

modal volume put qwen3-asr-ms-en \
  ./ms_en_lora \
  /checkpoints/ms_en_lora/final_adapter \
  --force
```

3. Run evaluation:

```bash
# Adapter-only WER
modal run train_modal.py \
  --action eval \
  --adapter /vol/checkpoints/ms_en_lora/final_adapter \
  --output-name ms_en_lora

# Base vs adapter + per-utterance compare
modal run train_modal.py \
  --action eval_compare \
  --adapter /vol/checkpoints/ms_en_lora/final_adapter \
  --output-name ms_en_lora
```

Expected test WER: about **8.66% / 5.29%** (no punct) with the adapter vs **13.37% / 6.62%** (no punct) for the base model.

---

## Part 2. Baseline inference benchmark

Single-stream inference for the **base** `Qwen3-ASR-1.7B` (no LoRA) on the uploaded test data. Each request processes **one** utterance (`batch_size=1`). We choose a representative collection of audio samples with different durations/bucket sizes (2–5 seconds, 5–15 seconds, 15–30 seconds).


```bash
modal run bench_modal.py \
  --per-bucket 25 \
  --batch-size 1 \
  --warmup 3 \
  --seed 42 \
  --output-name baseline_bench
```

Outputs:

- Volume: `/vol/checkpoints/baseline_bench/inference_baseline.json`
- Volume: `/vol/checkpoints/baseline_bench/inference_baseline_summary.json`
- Local mirror: `./inference_baseline_summary.json`

**RTF** = processing time / audio duration (lower is better).  
**Throughput** = total audio duration / processing time (audio-sec/s), or utterances / processing time (utt/s).

### Benchmark results (published run)

Environment: **NVIDIA L4** 24 GB, CUDA 12.4, PyTorch 2.5.1+cu124, bf16, SDPA, `batch_size=1`, warmup=3.

**Overall** (n=51 utterances, ~275 s audio):

| Metric | Value |
|--------|-------|
| avg RTF | 0.105 |
| p50 / p95 RTF | 0.104 / 0.135 |
| aggregate RTF | 0.099 |
| throughput | 10.10 audio-sec/s · 1.87 utt/s |

**By duration bucket:**

| Bucket | n | avg RTF | p50 | p95 | audio-sec/s |
|--------|---|--------|-----|-----|---------------|
| 2–5s | 25 | 0.115 | 0.118 | 0.142 | 8.75 |
| 5–15s | 25 | 0.097 | 0.099 | 0.118 | 10.48 |
| 15–30s | 1 | 0.044 | 0.044 | 0.044 | 22.98 |

**GPU during bench:** utilization avg ~55% (max 59%); torch peak allocated ~3.9 GB / reserved ~4.4 GB.

---

## Part 3/4/5. Inference Optimization + Concurrent Load Test + Optimization Journey.

In `loadtest_modal.py`, we send **multiple concurrent streams** against a **single** model instance and reports RTF, GPU utilization, VRAM, throughput, and WER. There's one audio clip per stream (`transcribe` batch size = concurrency).

**WER protocol:** a **fixed** set of **128** test clips (`--wer-set-size 128`, default `pool_size=128`). For concurrency `C`, clips are transcribed in batches of `min(C, 128)` until all 128 are covered-for example, `C=1` -> 128 batches of 1; `C=8` -> 16 batches of 8; `C=64` -> 2 batches of 64; `C=256` -> 1 batch of 128. We want to see if sending multiple concurrent streams harms the transcription accuracy or not.

For each concurrency level `C`, we run the experiment 4 times to average the results.


---

### Run 1 - Baseline PyTorch backend (`Qwen3ASRModel.from_pretrained`)

```bash
modal run loadtest_modal.py --max-concurrency 256 --wer-set-size 128
```

| Streams | Avg RTF | P50 RTF | P95 RTF | GPU avg % | GPU peak % | VRAM avg (MiB) | VRAM peak (MiB) | Throughput (audio-s/s) | WER % | WER no-punct % |
|--------:|--------:|--------:|--------:|----------:|-----------:|---------------:|----------------:|-----------------------:|------:|---------------:|
| 1 | 0.1799 | 0.1710 | 0.2415 | 35.2 | 40.0 | 4750 | 4753 | 5.97 | 8.05 | 5.56 |
| 2 | 0.1071 | 0.1023 | 0.1324 | 37.2 | 41.0 | 4770 | 4773 | 9.60 | 8.05 | 5.56 |
| 4 | 0.0592 | 0.0596 | 0.0648 | 38.1 | 52.0 | 4774 | 4795 | 16.88 | 8.13 | 5.64 |
| 8 | 0.0331 | 0.0333 | 0.0354 | 41.4 | 58.0 | 4737 | 4737 | 30.20 | 8.13 | 5.64 |
| 16 | 0.0215 | 0.0210 | 0.0241 | 44.0 | 86.0 | 5533 | 6457 | 46.58 | 8.22 | 5.73 |
| 32 | 0.0155 | 0.0150 | 0.0175 | 56.8 | 100.0 | 7377 | 8385 | 64.35 | 8.05 | 5.56 |
| 64 | 0.0156 | 0.0158 | 0.0162 | 52.6 | 100.0 | 7887 | 8385 | 64.02 | 8.05 | 5.56 |
| 128 | 0.0155 | 0.0154 | 0.0158 | 55.4 | 100.0 | 8134 | 8385 | 64.63 | 8.05 | 5.56 |
| 256 | 0.0156 | 0.0153 | 0.0164 | 55.1 | 100.0 | 8214 | 8385 | 64.23 | 8.05 | 5.56 |

Throughput plateaus near **~64 audio-sec/s** from 32 streams upward; higher concurrency does not increase throughput (the GPU is effectively saturated).

---

### Run 2 - vLLM backend

**Hypothesis:** we need a more efficient serving stack than the PyTorch path in Run 1. **vLLM** is built for LLM serving with **continuous batching**: on each decode step, sequences can join or leave the active batch as they finish, instead of waiting for an entire request group to complete. The PyTorch backend in this repo is closer to **static batching**-`N` clips are padded and processed together in one `transcribe` call, and the batch waits until the slowest sequence in that batch finishes. Under high concurrency, that mismatch shows up as early saturation (Run 1 plateaus near ~64 audio-sec/s) while the GPU still has headroom for better interleaving.

```bash
modal run loadtest_modal.py --backend vllm --wer-set-size 128 --max-concurrency 2048
```

| Streams | Avg RTF | P50 RTF | P95 RTF | GPU avg % | GPU peak % | VRAM avg (MiB) | VRAM peak (MiB) | Throughput (audio-s/s) | WER % | WER no-punct % |
|--------:|--------:|--------:|--------:|----------:|-----------:|---------------:|----------------:|-----------------------:|------:|---------------:|
| 1 | 0.0654 | 0.0652 | 0.0863 | 77.2 | 100.0 | 18920 | 18920 | 16.48 | 7.97 | 5.56 |
| 2 | 0.0331 | 0.0316 | 0.0407 | 86.6 | 100.0 | 19145 | 19146 | 31.09 | 8.30 | 5.81 |
| 4 | 0.0166 | 0.0167 | 0.0184 | 96.8 | 100.0 | 19146 | 19146 | 60.08 | 8.30 | 5.81 |
| 8 | 0.0090 | 0.0091 | 0.0096 | 100.0 | 100.0 | 19146 | 19146 | 110.19 | 8.30 | 5.81 |
| 16 | 0.0048 | 0.0049 | 0.0052 | 95.8 | 100.0 | 19146 | 19146 | 209.19 | 8.13 | 5.64 |
| 32 | 0.0034 | 0.0035 | 0.0036 | 98.0 | 100.0 | 19146 | 19146 | 291.66 | 8.30 | 5.81 |
| 64 | 0.0020 | 0.0020 | 0.0021 | 81.3 | 100.0 | 19146 | 19146 | 494.01 | 8.22 | 5.73 |
| 128 | 0.0015 | 0.0015 | 0.0015 | 68.4 | 100.0 | 19146 | 19146 | 682.17 | 8.13 | 5.64 |
| 256 | 0.0012 | 0.0012 | 0.0012 | 68.5 | 100.0 | 19146 | 19146 | 858.99 | 8.22 | 5.73 |
| 512 | 0.0011 | 0.0011 | 0.0011 | 69.9 | 100.0 | 19146 | 19146 | 935.02 | 8.05 | 5.56 |
| 1024 | 0.0010 | 0.0010 | 0.0010 | 68.1 | 100.0 | 19146 | 19146 | 964.14 | 8.13 | 5.64 |
| 2048 | 0.0010 | 0.0010 | 0.0010 | 67.6 | 100.0 | 19146 | 19146 | 970.66 | 8.05 | 5.56 |

Peak throughput in this run: **970.66 audio-sec/s** at 2048 concurrent streams. This was a significant improvement.

---

### Run 3 - vLLM with preloaded audio

**Hypothesis:** passing file paths forces disk I/O inside `transcribe` function of `Qwen3ASRModel` object; preloading audio numpy arrays into RAM (`--preload-audio`) should raise GPU utilization.

```bash
modal run loadtest_modal.py --backend vllm --wer-set-size 128 --max-concurrency 2048 --preload-audio
```

| Streams | Avg RTF | P50 RTF | P95 RTF | GPU avg % | GPU peak % | VRAM avg (MiB) | VRAM peak (MiB) | Throughput (audio-s/s) | WER % | WER no-punct % |
|--------:|--------:|--------:|--------:|----------:|-----------:|---------------:|----------------:|-----------------------:|------:|---------------:|
| 1 | 0.0615 | 0.0556 | 0.0904 | 98.0 | 100.0 | 18929 | 18929 | 17.16 | 8.22 | 5.73 |
| 2 | 0.0330 | 0.0315 | 0.0406 | 93.0 | 100.0 | 19152 | 19152 | 31.13 | 8.13 | 5.64 |
| 4 | 0.0165 | 0.0167 | 0.0180 | 97.2 | 100.0 | 19152 | 19152 | 60.39 | 8.13 | 5.64 |
| 8 | 0.0088 | 0.0089 | 0.0094 | 98.0 | 100.0 | 19152 | 19152 | 111.84 | 8.05 | 5.56 |
| 16 | 0.0046 | 0.0047 | 0.0050 | 96.5 | 100.0 | 19152 | 19152 | 218.86 | 8.13 | 5.64 |
| 32 | 0.0029 | 0.0029 | 0.0030 | 95.9 | 100.0 | 19152 | 19152 | 347.77 | 8.13 | 5.64 |
| 64 | 0.0018 | 0.0018 | 0.0019 | 96.7 | 100.0 | 19152 | 19152 | 565.07 | 8.05 | 5.56 |
| 128 | 0.0012 | 0.0012 | 0.0012 | 94.6 | 100.0 | 19152 | 19152 | 807.41 | 8.13 | 5.64 |
| 256 | 0.0009 | 0.0009 | 0.0009 | 85.1 | 100.0 | 19152 | 19152 | 1077.58 | 8.22 | 5.73 |
| 512 | 0.0008 | 0.0008 | 0.0008 | 91.3 | 100.0 | 19152 | 19152 | 1187.54 | 8.22 | 5.73 |
| 1024 | 0.0008 | 0.0008 | 0.0009 | 87.5 | 100.0 | 19152 | 19152 | 1206.82 | 8.22 | 5.73 |
| 2048 | 0.0008 | 0.0008 | 0.0008 | 84.6 | 100.0 | 19152 | 19152 | 1257.21 | 8.22 | 5.73 |

Peak throughput: **1257.21 audio-sec/s** at 2048 streams.

---

### Run 4 - vLLM with ASR-oriented `max_new_tokens` / `max_model_len`

`loadtest_modal.py` exposes vLLM’s `max_new_tokens` and `max_model_len` (defaults in the script are **8192** and **65536**, which are reasonable for long chat generations but oversized for ASR). For transcription, the thinker only needs enough context for the audio encoder output plus a **short** text decode-typically tens to low hundreds of tokens, not thousands. Leaving chat-scale limits in place reserves a large **KV cache** per slot and can cap how many concurrent sequences fit in VRAM even when outputs stay short.

Run 4 sets **ASR-oriented** caps: `max_new_tokens=512`, `max_model_len=8192`. That is a safety margin above real transcript length while shrinking per-request memory so more streams can share the same ~19 GiB footprint you see in Runs 2–3.

```bash
modal run loadtest_modal.py --backend vllm --wer-set-size 128 --max-concurrency 2048 --preload-audio \
  --max-new-tokens 512 --max-model-len 8192
```

| Streams | Avg RTF | P50 RTF | P95 RTF | GPU avg % | GPU peak % | VRAM avg (MiB) | VRAM peak (MiB) | Throughput (audio-s/s) | WER % | WER no-punct % |
|--------:|--------:|--------:|--------:|----------:|-----------:|---------------:|----------------:|-----------------------:|------:|---------------:|
| 1 | 0.0610 | 0.0557 | 0.0885 | 78.2 | 100.0 | 18920 | 18920 | 14.67 | 7.97 | 5.56 |
| 2 | 0.0332 | 0.0317 | 0.0407 | 91.4 | 100.0 | 19145 | 19146 | 29.72 | 8.30 | 5.81 |
| 4 | 0.0167 | 0.0167 | 0.0188 | 98.0 | 100.0 | 19146 | 19146 | 59.63 | 8.30 | 5.81 |
| 8 | 0.0088 | 0.0089 | 0.0094 | 100.0 | 100.0 | 19146 | 19146 | 110.97 | 8.30 | 5.81 |
| 16 | 0.0046 | 0.0047 | 0.0050 | 97.0 | 100.0 | 19146 | 19146 | 218.98 | 8.22 | 5.73 |
| 32 | 0.0029 | 0.0029 | 0.0031 | 92.0 | 100.0 | 19146 | 19146 | 344.79 | 8.30 | 5.81 |
| 64 | 0.0018 | 0.0018 | 0.0019 | 90.4 | 100.0 | 19146 | 19146 | 565.52 | 8.13 | 5.64 |
| 128 | 0.0012 | 0.0012 | 0.0012 | 90.1 | 100.0 | 19146 | 19146 | 807.30 | 8.22 | 5.73 |
| 256 | 0.0009 | 0.0009 | 0.0009 | 84.4 | 100.0 | 19146 | 19146 | 1073.08 | 8.13 | 5.64 |
| 512 | 0.0009 | 0.0009 | 0.0009 | 87.0 | 100.0 | 19146 | 19146 | 1163.64 | 8.30 | 5.81 |
| 1024 | 0.0008 | 0.0008 | 0.0008 | 84.5 | 100.0 | 19146 | 19146 | 1210.42 | 8.13 | 5.64 |
| 2048 | 0.0008 | 0.0008 | 0.0008 | 84.8 | 100.0 | 19146 | 19146 | 1270.23 | 8.22 | 5.73 |

Results are similar to Run 3, so we did not achieve any improvement. Profiling (`modal run loadtest_modal.py --torch-profile --backend vllm --profile-streams 256`) shows top CUDA time in compiled FX graphs and `aten::mm` / CUTLASS GEMM (~40–45%), many `execute_context_*_generation_*` decode steps, FlashAttention, and `aten::cudnn_convolution` ~7% (audio encoder). The bottleneck is GPU compute (LM matmuls, decode, encoder conv), not disk prep-further gains likely need faster compute (quantization, shorter decode, or dedicated serving), not more preload or `max_model_len` tuning alone.
Profiling result is at [profiler_out_0.txt](vllm_torch_profile/loadtest_vllm_torch_profile_20261013_071106/traces/profiler_out_0.txt)

---

### Run 5 - vLLM + FP8 (ModelOpt, MS-EN calibration)

We run quantization, resulting weights are stored on the volume (Malaysian English `train.jsonl`, adapted from [VinRobotics qwen3-asr/vllm](https://github.com/VinRobotics/model-quantization-recipes/tree/main/recipes/qwen3-asr/vllm) and [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt/torch/quantization)):

```bash
modal run quantize_modal.py --format fp8
# optional: modal run quantize_modal.py --format nvfp4
```

Load test (exports at `/vol/checkpoints/qwen3asr_ms_en_fp8_vllm`):

```bash
modal run loadtest_modal.py --backend vllm --quantization fp8 \
  --wer-set-size 128 --max-concurrency 2048 --preload-audio
```

| Streams | Avg RTF | P50 RTF | P95 RTF | GPU avg % | GPU peak % | VRAM avg (MiB) | VRAM peak (MiB) | Throughput (audio-s/s) | WER % | WER no-punct % |
|--------:|--------:|--------:|--------:|----------:|-----------:|---------------:|----------------:|-----------------------:|------:|---------------:|
| 1 | 0.0371 | 0.0378 | 0.0452 | 94.7 | 100.0 | 18866 | 18866 | 29.11 | 7.97 | 5.39 |
| 2 | 0.0232 | 0.0222 | 0.0283 | 77.0 | 100.0 | 19089 | 19089 | 44.29 | 7.80 | 5.31 |
| 4 | 0.0112 | 0.0112 | 0.0122 | 98.0 | 100.0 | 19091 | 19091 | 89.46 | 7.80 | 5.31 |
| 8 | 0.0061 | 0.0062 | 0.0065 | 99.2 | 100.0 | 19091 | 19091 | 163.22 | 8.30 | 5.81 |
| 16 | 0.0034 | 0.0035 | 0.0038 | 99.8 | 100.0 | 19091 | 19091 | 292.47 | 8.30 | 5.81 |
| 32 | 0.0021 | 0.0021 | 0.0022 | 86.0 | 100.0 | 19091 | 19091 | 472.30 | 7.88 | 5.39 |
| 64 | 0.0014 | 0.0014 | 0.0015 | 88.3 | 100.0 | 19091 | 19091 | 714.30 | 7.88 | 5.39 |
| 128 | 0.0010 | 0.0010 | 0.0011 | 80.9 | 100.0 | 19091 | 19091 | 953.82 | 8.05 | 5.56 |
| 256 | 0.0008 | 0.0008 | 0.0008 | 88.5 | 100.0 | 19091 | 19091 | 1251.98 | 8.05 | 5.56 |
| 512 | 0.0007 | 0.0007 | 0.0007 | 91.6 | 100.0 | 19091 | 19091 | 1407.46 | 8.05 | 5.56 |
| 1024 | 0.0007 | 0.0007 | 0.0007 | 83.7 | 100.0 | 19091 | 19091 | 1472.89 | 8.05 | 5.56 |
| 2048 | 0.0007 | 0.0007 | 0.0007 | 82.2 | 100.0 | 19091 | 19091 | 1499.82 | 8.05 | 5.56 |

Peak throughput: **1499.82 audio-sec/s** at 2048 streams. So quantization indeed helped.

---

### Run 6 - vLLM + NVFP4 (ModelOpt, MS-EN calibration)

Requires NVFP4 export on the volume (`modal run quantize_modal.py --format nvfp4`). Checkpoint: `/vol/checkpoints/qwen3asr_ms_en_nvfp4_vllm`.

```bash
modal run loadtest_modal.py --backend vllm --quantization nvfp4 \
  --wer-set-size 128 --max-concurrency 2048 --preload-audio
```

| Streams | Avg RTF | P50 RTF | P95 RTF | GPU avg % | GPU peak % | VRAM avg (MiB) | VRAM peak (MiB) | Throughput (audio-s/s) | WER % | WER no-punct % |
|--------:|--------:|--------:|--------:|----------:|-----------:|---------------:|----------------:|-----------------------:|------:|---------------:|
| 1 | 0.0459 | 0.0279 | 0.0964 | 54.0 | 66.0 | 18873 | 18873 | 28.89 | 8.13 | 5.48 |
| 2 | 0.0185 | 0.0182 | 0.0211 | 88.7 | 100.0 | 19096 | 19096 | 55.24 | 8.13 | 5.39 |
| 4 | 0.0085 | 0.0085 | 0.0093 | 97.7 | 100.0 | 19096 | 19096 | 117.74 | 8.22 | 5.48 |
| 8 | 0.0046 | 0.0046 | 0.0050 | 100.0 | 100.0 | 19096 | 19096 | 217.62 | 8.13 | 5.48 |
| 16 | 0.0025 | 0.0026 | 0.0028 | 91.0 | 94.0 | 19096 | 19096 | 395.18 | 8.05 | 5.39 |
| 32 | 0.0017 | 0.0017 | 0.0017 | 93.8 | 100.0 | 19096 | 19096 | 589.93 | 8.22 | 5.39 |
| 64 | 0.0012 | 0.0012 | 0.0013 | 83.3 | 100.0 | 19096 | 19096 | 822.97 | 8.22 | 5.39 |
| 128 | 0.0010 | 0.0010 | 0.0010 | 88.6 | 100.0 | 19096 | 19096 | 1015.45 | 8.30 | 5.48 |
| 256 | 0.0008 | 0.0008 | 0.0008 | 78.9 | 100.0 | 19096 | 19096 | 1273.63 | 8.30 | 5.48 |
| 512 | 0.0007 | 0.0007 | 0.0008 | 84.3 | 100.0 | 19096 | 19096 | 1355.89 | 8.22 | 5.48 |
| 1024 | 0.0007 | 0.0007 | 0.0007 | 81.5 | 100.0 | 19096 | 19096 | 1406.91 | 8.13 | 5.39 |
| 2048 | 0.0007 | 0.0007 | 0.0007 | 83.6 | 100.0 | 19096 | 19096 | 1450.25 | 8.13 | 5.39 |

Peak throughput: **1450.25 audio-sec/s** at 2048 streams.

**Wrap-up:** starting from ~**64** audio-sec/s on the PyTorch backend (Run 1), vLLM continuous batching lifted peak throughput to ~**971** audio-sec/s (Run 2). Preloading audio removed CPU/disk stalls and reached ~**1257** audio-sec/s (Run 3). Tighter ASR generation limits (Run 4) did not materially change that story; profiling pointed at GPU matmuls and decode as the main cost. **FP8** ModelOpt weights on MS-EN calibration gave the best result in this ladder-~**1500** audio-sec/s at 2048 streams (Run 5)-with **NVFP4** close behind at ~**1450** audio-sec/s (Run 6). Across these runs, WER on the fixed 128-clip set stayed in the same band as the LoRA fine-tuned baseline (~8% / ~5.5% no-punct), so the throughput gains did not come at the cost of obvious accuracy collapse under concurrency.


## Part 6: Bottleneck Analysis

The final stack is vLLM + FP8 (ModelOpt, MS-EN calibration) + preloaded audio on a single Modal L4 (24 GB). Instrumentation:

| Tool | What it measured |
|------|------------------|
| `nvidia-smi` poller (250 ms) | `utilization.gpu`, `memory.used` during each concurrency wave |
| Custom wave timers | wall time, per-wave RTF (`proc / total_audio`), throughput (audio-sec/s) |
| `--cpu-prep-profile` | `transcribe(paths)` vs `transcribe(numpy)` + cold/warm `sf.read` |
| vLLM / PyTorch Profiler (`--torch-profile`, 256 streams) | CUDA kernel table (`aten::mm`, CUTLASS, FlashAttention, conv, HtoD) |

RTF in this harness is batch processing time / total audio in the wave (same value for every clip in a wave). That is why P95 RTF falls as concurrency rises: more audio is packed into one GPU wave. P95 RTF stayed far below 0.5 at every level we ran, including 2048 streams.

### What is *not* the limiter (anymore)

- CPU preprocessing / disk I/O. Run 2 (path-based `transcribe`) peaked at ~971 audio-sec/s with GPU avg falling to ~68% above 128 streams. Run 3 (preload wavs to RAM) recovered GPU avg into the mid-80s–90s and lifted the plateau to ~1257 audio-sec/s. After preload, disk decode is off the timed path.
- Static batching. The PyTorch backend plateaued at ~64 audio-sec/s once GPU peak hit 100% (Run 1, C≥32). vLLM continuous batching removed that scheduling mismatch.

### What actually limits further concurrency / throughput

GPU compute on the autoregressive decoder, with a side of diminishing batching returns.

PyTorch Profiler (vLLM, 256 preloaded streams) CUDA share:

| Kernel / group | Self CUDA | Role |
|----------------|----------:|------|
| Compiled FX graph + `aten::mm` / CUTLASS bf16 GEMM | ~40–45% | LM projections |
| `execute_context_*_generation_*` | many long sequential steps | token-by-token decode |
| FlashAttention (`flash_fwd_splitkv`, `_vllm_fa2`) | ~8–12% | attention |
| `aten::cudnn_convolution` | ~7% | audio encoder |
| `aten::copy_` / elementwise / RMSNorm | small | not the ceiling |

So the critical path is matmul + decode steps, then encoder conv. That matches the load-test shape of the final run (Run 5, FP8):

| Streams | P95 RTF | GPU avg % | GPU peak % | VRAM peak (MiB) | Throughput (audio-s/s) |
|--------:|--------:|----------:|-----------:|----------------:|-----------------------:|
| 256 | 0.0008 | 88.5 | 100 | 19091 | 1252 |
| 512 | 0.0007 | 91.6 | 100 | 19091 | 1407 |
| 1024 | 0.0007 | 83.7 | 100 | 19091 | 1473 |
| 2048 | 0.0007 | 82.2 | 100 | 19091 | 1500 |

From 256 -> 2048 streams, throughput only grows ~20% while GPU average utilization slips from the high 80s toward ~82% (peak still 100%). Extra streams no longer feed useful work into the GPU: the decoder is already issuing a long chain of small GEMMs, and the L4 is compute-saturated on those kernels. FP8 helped (Run 3 ~1257 -> Run 5 ~1500) because it made those matmuls cheaper; NVFP4 on L4 did not beat FP8 (Ada FP8 tensor cores vs. NVFP4, which is a better fit for newer GPUs).

Tighter `max_new_tokens` / `max_model_len` (Run 4) did not help, which is consistent with a compute bottleneck rather than a KV-capacity bottleneck at these short ASR outputs.

Bottom line: concurrency in the final implementation is limited by GPU compute (LM GEMM + autoregressive decode) on one L4. So the next step would be faster compute (more advanced GPU or more GPUs).


## Conclusion

What was the maximum sustainable concurrency achieved while maintaining P95 RTF ≤ 0.5?  
2048 concurrent streams on the final stack (vLLM + FP8 + preload), with P95 RTF 0.0007.

What was the main bottleneck preventing further concurrency?  
GPU compute on decoder GEMMs and token-by-token generation (Profiler: ~40–45% CUDA in `aten::mm` / CUTLASS / compiled FX, plus a long `execute_context_*_generation_*` tail). Throughput flattens near ~1500 audio-sec/s above ~512 streams while nvidia-smi still shows 100% peak util and a pre-allocated ~19.1 GiB working set.

Which optimization produced the largest improvement?  
Switching from the PyTorch `transcribe` backend to vLLM continuous batching (Run 1 -> Run 2): peak throughput ~64 -> ~971 audio-sec/s (~15×).

If given another week, what would you try next to increase concurrent streams?  
1. Move off L4 to an FP8-stronger card (L40S / H100).  
2. Other quantization techniques to produce int8 or int4 models.

If this system needed to support 1,000 simultaneous ASR sessions, how would you design the GPU deployment architecture?  

From the experiments, one single L4 already sustains 1000+ concurrent batch streams at P95 RTF << 0.5 (~1500 audio-sec/s with vLLM+FP8). For production deployment with 1000 sessions, I would design the system like this:

- CPU edge (FastAPI): auth, session IDs, upload/download, request validation, queueing/backpressure. Keep decode/resample/VAD if needed here so GPUs only see ready audio and do the main ASR heavy work.
- GPU workers: one vLLM + FP8 replica per GPU; scale replicas horizontally (can use Modal).
- Path: client -> FastAPI -> queue -> GPU worker -> transcript back via session. Autoscale on queue depth / P95 latency; canary one replica for WER before rolling weights.