# TorchTPU Document OCR & Visual Parsing Serving on Cloud TPU v7x

High-throughput document OCR and structured Markdown extraction benchmarks on **Google Cloud TPU v7x** using `vllm-torchtpu v0.33.0`. On a single TPU v7x chip (`TP=2`), `deepseek-ai/DeepSeek-OCR` sustains **747.4 pages/min** (`12.46 pages/sec`, `2,614 tok/s`, **100.0%** field accuracy) on single-view `640x640` invoices and **655.4 pages/min** (`2,840 tok/s`, **100.0%** accuracy) on tables, while `Qwen/Qwen2.5-VL-7B-Instruct` sustains **359.8 pages/min** (`2,556 tok/s`, `25.4 ms` `c=1` TTFT, **100.0%** accuracy) on full `1024x1400` invoices and **404.0 pages/min** on dense financial tables.

---

## Models, Framework & Hardware

| Model | Role | Architecture | Params / Precision | Input Vision Budget | License |
|---|---|---|---|---|---|
| [`deepseek-ai/DeepSeek-OCR`](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | Dedicated optical-compression OCR | DeepEncoder (SAM + CLIP) + DeepSeek-3B MoE decoder | ~3B MoE / `bf16` | **288 tokens** (`640x640` single-view) | MIT |
| [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | General-purpose document VLM | Dynamic-resolution ViT + Qwen2.5-7B decoder | 8.3B / `bf16` | **1,889 tokens** (`1024x1400`) / **4,054 tokens** (`1536x2048`) | Apache-2.0 |

* **Hardware:** Google Cloud TPU v7x (2 TensorCores per chip, 192 GB HBM per chip). Primary serving runs at **`TP=2` (1 TPU v7x chip)**; scaling comparison also evaluated at **`TP=4` (2 TPU v7x chips)**.
* **Software stack:** PyTorch + TorchTPU (`torch_tpu`), `vllm-torchtpu v0.33.0` OpenAI-compatible `/v1/chat/completions` streaming server. Also audited on `sglang-torchtpu v0.2.0` (both `DeepSeek-OCR` and `Qwen2.5-VL-7B-Instruct` fail during multimodal image placement at `sglang_tpu/mm_embed_tpu.py:139`; full trace in [APPENDIX.md](APPENDIX.md)).

---

## Main Results

### 1. `deepseek-ai/DeepSeek-OCR` — 1 TPU v7x Chip (`TP=2`, Single-View `640x640`)

Measured warm steady-state streaming HTTP throughput (`results/v7x_deepseek_ocr_tp2_singleview.jsonl`), using `--max-model-len 4096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0`:

| Document Profile | Concurrency | Input / Output Toks | TTFT p50 | Page E2E p50 | Throughput (`pages/min`) | Output Rate (`tok/s`) | Field Match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Invoice (`640x640`) | 1 | 288 / 210 | **67.3 ms** | 0.67 s | 88.9 | 311.1 | 100.0% |
| Invoice (`640x640`) | 4 | 288 / 210 | 149.8 ms | 0.82 s | 289.5 | 1,013.3 | 100.0% |
| Invoice (`640x640`) | 8 | 288 / 210 | 315.9 ms | 0.97 s | 504.2 | 1,764.8 | 100.0% |
| Invoice (`640x640`) | 16 | 288 / 210 | 554.9 ms | 1.27 s | **747.4** (`12.46/s`) | 2,614.4 | **100.0%** |
| Financial Table (`640x640`) | 1 | 288 / 260 | **66.0 ms** | 0.82 s | 74.0 | 320.5 | 100.0% |
| Financial Table (`640x640`) | 4 | 288 / 260 | 145.4 ms | 0.92 s | 259.2 | 1,123.4 | 100.0% |
| Financial Table (`640x640`) | 8 | 288 / 260 | 278.6 ms | 1.11 s | 433.9 | 1,880.3 | 100.0% |
| Financial Table (`640x640`) | 16 | 288 / 260 | 549.4 ms | 1.45 s | **655.4** (`10.92/s`) | **2,840.1** | **100.0%** |

* **Operational constraint for DeepSeek-OCR on TPU v7x:** Serving at `TP=2` resolves the `TP=1` VMEM overflow (`79.89 MB > 63.94 MB`), and keeping page dimensions `<= 640x640` uses the native single-view encoder grid cleanly without invoking multi-crop bicubic position-embedding interpolation (full diagnostic in [APPENDIX.md](APPENDIX.md)).

### 2. `Qwen/Qwen2.5-VL-7B-Instruct` — Full-Page & High-Res OCR (`TP=2` vs `TP=4`)

Measured warm steady-state streaming HTTP throughput (`results/v7x_qwen25_vl_7b_ocr_tp2.jsonl`, `results/v7x_qwen25_vl_7b_ocr_tp4.jsonl`):

| Config | Document Profile | Concurrency | Input / Output Toks | TTFT p50 | Throughput (`pages/min`) | Output Rate (`tok/s`) | Field Match |
|---|---|---:|---:|---:|---:|---:|---:|
| **`TP=2` (1 chip)** | Invoice (`1024x1400`) | 1 | 1,889 / 426 | **25.4 ms** | 41.9 | 297.8 | 100.0% |
| **`TP=2` (1 chip)** | Invoice (`1024x1400`) | 8 | 1,889 / 426 | 70.4 ms | 275.2 | 1,954.8 | 100.0% |
| **`TP=2` (1 chip)** | Invoice (`1024x1400`) | 16 | 1,889 / 426 | 450.7 ms | **359.8** (`6.00/s`) | **2,556.2** | **100.0%** |
| **`TP=2` (1 chip)** | Table (`1024x1400`) | 1 | 1,889 / 338 | **25.6 ms** | 52.6 | 296.4 | 83.3% |
| **`TP=2` (1 chip)** | Table (`1024x1400`) | 16 | 1,889 / 338 | 255.1 ms | **404.0** (`6.73/s`) | 2,276.7 | 83.3% |
| **`TP=2` (1 chip)** | High-Res Invoice (`1536x2048`) | 4 | 4,054 / 411 | **59.8 ms** | **145.1** (`2.42/s`) | 994.2 | **100.0%** |
| `TP=4` (2 chips) | Invoice (`1024x1400`) | 1 | 1,889 / 426 | 26.1 ms | 42.9 | 304.8 | 100.0% |
| `TP=4` (2 chips) | Invoice (`1024x1400`) | 16 | 1,889 / 426 | 843.0 ms | **397.3** (`6.62/s`) | **2,821.2** | **100.0%** |

* **Chip efficiency comparison (`TP=2` vs `TP=4`):** On `1024x1400` invoices at `c=16`, `TP=2` (1 chip) delivers **359.8 pages/min per chip** vs **198.7 pages/min per chip** at `TP=4` (2 chips, `397.3` total) — **`1.81x` higher throughput per chip** for independent `TP=2` data-parallel replicas.

---

## How to Reproduce

### `deepseek-ai/DeepSeek-OCR` (`TP=2`, 1 TPU v7x Chip)

```bash
# Terminal 1: Start DeepSeek-OCR server (TP=2 fits within TPU v7x VMEM)
vllm serve deepseek-ai/DeepSeek-OCR \
  --tensor-parallel-size 2 --max-model-len 4096 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000

# Terminal 2: Run crisp single-view 640x640 OCR benchmark
python3 scripts/benchmark_document_ocr.py \
  --model deepseek-ai/DeepSeek-OCR --profile-set singleview-640 \
  --base-url http://127.0.0.1:8000 --tp 2
```

### `Qwen/Qwen2.5-VL-7B-Instruct` (`TP=2`, 1 TPU v7x Chip)

```bash
# Terminal 1: Start Qwen2.5-VL-7B-Instruct server
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 2 --max-model-len 8192 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --trust-remote-code --port 8001

# Terminal 2: Run full-page 1024x1400 and 1536x2048 OCR benchmark
python3 scripts/benchmark_document_ocr.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct --profile-set full \
  --base-url http://127.0.0.1:8001 --tp 2
```

---

## Full Data & Engineering Notes

* [APPENDIX.md](APPENDIX.md) — `DeepSeek-OCR` `TP=1` VMEM analysis, multi-crop `aten::_upsample_bicubic2d_aa.out` diagnostic, cold vs warm vision-batch compile measurements, `Qwen2.5-VL-7B` `TP=2` vs `TP=4` stability notes, and `sglang-torchtpu v0.2.0` audit traces.
* [WORKING_LOG.md](WORKING_LOG.md) — Chronological engineering bring-up and dual-engine audit log.
* [`results/`](results/) — Raw verified JSONL measurements backing every table above.
