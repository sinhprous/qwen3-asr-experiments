# Qwen3-ASR Malaysian English fine-tune

LoRA fine-tune of [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on Common Voice Malaysian English, trained on Modal (L4).

| Artifact | Location |
|----------|----------|
| Published LoRA adapter | [`sinhprous/qwen3-asr-1.7b-ms-en-lora`](https://huggingface.co/sinhprous/qwen3-asr-1.7b-ms-en-lora) |
| Modal volume name (in scripts) | `qwen3-asr-ms-en` |
| Training run name | `ms_en_lora_bs_4_acc_4` |

Use **your own Modal profile / workspace**. Activate it before any `modal` command:

```bash
modal profile activate <your-profile>
# or: export MODAL_PROFILE=<your-profile>
```

---

## Setup

```bash
pip install -r requirements.txt
pip install modal huggingface_hub
modal profile activate <your-profile>
```

Optional: copy `.env.example` → `.env` and set `HF_TOKEN=...` for Hugging Face downloads. Do not commit `.env`.

**Data.** Prepare Common Voice Malaysian English (folder with `train/dev/test.tsv` + `clips/`):

```bash
python prep_data.py \
  --src "/path/to/cv26-malaysian-english" \
  --out ./data_ms_en
```

Writes `data_ms_en/{train,dev,test}.jsonl` and `data_ms_en/wavs/` (16 kHz mono). Expected sizes for the published run: **2343 / 62 / 71** train/dev/test utterances.

Upload to **your** Modal volume:

```bash
modal run train_modal.py --action upload --local-dir ./data_ms_en
```

---

## Reproduce training

Hyperparameters for `ms_en_lora_bs_4_acc_4` (defaults in `train_modal.py`):

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen3-ASR-1.7B` |
| GPU | Modal **L4** |
| LoRA | r=16, α=32, dropout=0.05, `target_modules="all-linear"` |
| Train batch | `per_device_train_batch_size=4`, `gradient_accumulation_steps=4` (effective **16**) |
| LR | 2e-5, cosine, warmup_ratio=0.03 |
| Max epochs | 10, early stopping patience=4 on `eval_loss` |
| Precision | bfloat16 on L4 |

```bash
modal run train_modal.py \
  --action train \
  --output-name ms_en_lora_bs_4_acc_4
```

Pipeline: baseline test WER → LoRA train → post test WER. Artifacts on your volume:

```
/vol/checkpoints/ms_en_lora_bs_4_acc_4/
  final_adapter/
  wer_before.json
  wer_after.json
  wer_per_utt_compare.json
  summary.json
  checkpoint-*
```

### WER protocol

Eval reports **two** corpus WERs (same hypotheses, different text normalization):

| Metric | Normalization before `jiwer.process_words` |
|--------|--------------------------------------------|
| **WER** | strip `<asr_text>` → lowercase → **keep punctuation** → collapse whitespace |
| **WER (no punct)** | same, but also remove `string.punctuation` |

Empty references (after punct-kept normalize) are skipped. Sanity-check: `python test_wer.py --self-test`.

### Training results (published run)

Test set **n=71** (both metrics on the same saved hypotheses):

| Model | WER | Sub / Del / Ins | Hits | WER (no punct) | Sub / Del / Ins | Hits |
|-------|-----|-----------------|------|----------------|-----------------|------|
| Base `Qwen3-ASR-1.7B` | **13.37%** | 65 / 4 / 2 | 462 | **6.62%** | 31 / 2 / 2 | 496 |
| + LoRA (`ms_en_lora_bs_4_acc_4`) | **8.66%** | 41 / 4 / 1 | 486 | **5.29%** | 25 / 2 / 1 | 502 |
| Δ | **−4.71 abs** | | | **−1.32 abs** | | |

Per-utterance compare:

- punct kept: **22 improved**, **2 worsened**, **47 unchanged**
- no punct: **6 improved**, **1 worsened**, **64 unchanged**

---

## Reproduce eval (Hugging Face adapter)

Published weights are at [`sinhprous/qwen3-asr-1.7b-ms-en-lora`](https://huggingface.co/sinhprous/qwen3-asr-1.7b-ms-en-lora). Reproduce WER on your own Modal account:

1. Prepare and upload data (see Setup).
2. Download the adapter and put it on your volume:

```bash
huggingface-cli download sinhprous/qwen3-asr-1.7b-ms-en-lora \
  --local-dir ./ms_en_lora_bs_4_acc_4

modal volume put qwen3-asr-ms-en \
  ./ms_en_lora_bs_4_acc_4 \
  /checkpoints/ms_en_lora_bs_4_acc_4/final_adapter \
  --force
```

3. Run eval:

```bash
# Adapter-only WER
modal run train_modal.py \
  --action eval \
  --adapter /vol/checkpoints/ms_en_lora_bs_4_acc_4/final_adapter \
  --output-name ms_en_lora_bs_4_acc_4

# Base vs adapter + per-utt compare
modal run train_modal.py \
  --action eval_compare \
  --adapter /vol/checkpoints/ms_en_lora_bs_4_acc_4/final_adapter \
  --output-name ms_en_lora_bs_4_acc_4
```

Expect test WER near **8.66% / 5.29% no-punct** (adapter) vs **13.37% / 6.62% no-punct** (base).

---

## Reproduce inference benchmark (`bench_modal.py`)

Baselines **original** `Qwen3-ASR-1.7B` (no LoRA) on your uploaded data.

```bash
modal run bench_modal.py \
  --per-bucket 25 \
  --batch-size 1 \
  --warmup 3 \
  --seed 42 \
  --output-name baseline_bench
```

Writes:

- Volume: `/vol/checkpoints/baseline_bench/inference_baseline.json`
- Volume: `/vol/checkpoints/baseline_bench/inference_baseline_summary.json`
- Local mirror: `./inference_baseline_summary.json`

RTF = processing time / audio duration (lower is better).

### Benchmark results (published run)

Environment: **NVIDIA L4** 22.5 GB, CUDA 12.4, PyTorch 2.5.1+cu124, bf16, SDPA, `batch_size=1`, warmup=3.

**Overall** (n=51 utterances, ~275 s audio):

| Metric | Value |
|--------|-------|
| avg RTF | 0.105 |
| p50 / p95 RTF | 0.104 / 0.135 |
| aggregate RTF | 0.099 |
| throughput | 10.10 audio-sec/sec · 1.87 utt/sec |

**By duration bucket:**

| Bucket | n | avg RTF | p50 | p95 | audio-sec/sec |
|--------|---|--------|-----|-----|---------------|
| 2–5s | 25 | 0.115 | 0.118 | 0.142 | 8.75 |
| 5–15s | 25 | 0.097 | 0.099 | 0.118 | 10.48 |
| 15–30s | 1 | 0.044 | 0.044 | 0.044 | 22.98 |

**GPU during bench:** util avg ~55% (max 59%), torch peak allocated ~3.9 GB / reserved ~4.4 GB.

---

## Reproduce load test

```
>> modal run loadtest_modal.py --waves 4 --max-concurrency 256 --pool-size 128 --wer-set-size 128
```

waves = 4 --> run each n-concurrent streams 4 times and average results
wer_set_size --> fixed set to compute WER, e.g: 
fixed set size 64

C=1 → 64 batches of 1
C=8 → 8 batches of 8
C=64 → 1 batch of 64
C=256 → need padding or skip WER for C > 64; or use set size ≥ max concurrency (e.g. 256)

=== Load test results (baseline) ===
```
Concurrent Streams | Avg RTF | P50 RTF | P95 RTF | GPU Util % | VRAM (MiB) | Throughput (audio-s/s) | WER % | WER no-punct %
-------------------+---------+---------+---------+------------+------------+------------------------+-------+---------------
1                  | 0.1614  | 0.1605  | 0.1708  | 39.2       | 4745       | 6.20                   | 7.72  | 5.15          
2                  | 0.1030  | 0.1010  | 0.1181  | 36.0       | 4764       | 9.73                   | 7.66  | 5.02          
4                  | 0.0500  | 0.0531  | 0.0590  | 39.7       | 4789       | 20.15                  | 7.66  | 5.02          
8                  | 0.0309  | 0.0317  | 0.0331  | 46.9       | 4737       | 32.32                  | 7.72  | 5.09          
16                 | 0.0189  | 0.0194  | 0.0205  | 50.1       | 5776       | 52.74                  | 7.72  | 5.09          
32                 | 0.0123  | 0.0124  | 0.0135  | 59.5       | 6733       | 81.05                  | 7.79  | 5.09          
64                 | 0.0122  | 0.0122  | 0.0129  | 58.1       | 8306       | 81.74                  | 7.79  | 5.09          
128                | 0.0122  | 0.0122  | 0.0123  | 58.7       | 8684       | 82.02                  | 7.79  | 5.09          
256                | 0.0123  | 0.0122  | 0.0124  | 57.9       | 8868       | 81.62                  | 7.79  | 5.09         
```

```
=== Concurrent stream ladder (baseline) ===
Concurrent Streams | Avg RTF | P50 RTF | P95 RTF | GPU avg % | GPU peak % | VRAM avg (MiB) | VRAM peak (MiB) | Throughput 
(audio-s/s) | WER % | WER no-punct %
-------------------+---------+---------+---------+-----------+------------+----------------+-----------------+--------------------
----+-------+---------------
1                  | 0.1141  | 0.1146  | 0.1175  | 55.3      | 56.0       | 4745           | 4747            | 8.77               
| —     | —             
2                  | 0.0736  | 0.0710  | 0.0846  | 51.7      | 55.0       | 4764           | 4771            | 13.63              
| —     | —             
4                  | 0.0355  | 0.0375  | 0.0418  | 56.2      | 76.0       | 4788           | 4793            | 28.37              
| —     | —             
8                  | 0.0223  | 0.0226  | 0.0239  | 62.4      | 90.0       | 4737           | 4737            | 44.76              
| —     | —             
16                 | 0.0140  | 0.0143  | 0.0151  | 62.4      | 100.0      | 5759           | 6065            | 71.17              
| —     | —             
32                 | 0.0094  | 0.0095  | 0.0102  | 71.4      | 100.0      | 6739           | 7773            | 106.22             
| —     | —             
64                 | 0.0093  | 0.0093  | 0.0096  | 78.2      | 100.0      | 8296           | 9073            | 107.66             
| —     | —             
128                | 0.0093  | 0.0093  | 0.0094  | 76.9      | 100.0      | 8676           | 9073            | 107.11             
| —     | —             
256                | 0.0094  | 0.0094  | 0.0094  | 77.2      | 100.0      | 8862           | 9073            | 106.51             
| —     | — 
```

---

## Project layout

| File | Role |
|------|------|
| `prep_data.py` | Local Common Voice → JSONL + 16 kHz WAVs |
| `train_modal.py` | Modal upload / train / eval / eval_compare |
| `bench_modal.py` | Modal inference RTF / throughput baseline |
| `loadtest_modal.py` | Modal concurrent-stream load test (+ WER per level) |
| `test_wer.py` | Isolated WER protocol check (`--self-test` / `--ref` / `--hyp`) |
| `requirements.txt` | Local prep deps |
| `.env.example` | `HF_TOKEN=` placeholder |
