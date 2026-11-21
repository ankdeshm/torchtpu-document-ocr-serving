# Appendix: TPU VMEM Tuning, Vision-Op Diagnostics & Full Sweeps

This appendix documents the engineering progression required to unlock `deepseek-ai/DeepSeek-OCR` on TPU v7x under `vllm-torchtpu v0.33.0`, warm vs cold vision-batch compilation effects, and `Qwen/Qwen2.5-VL-7B-Instruct` tensor-parallelism behavior.

---

## 1. Unlocking `deepseek-ai/DeepSeek-OCR` on TPU v7x (`vllm-torchtpu v0.33.0`)

Enabling `deepseek-ai/DeepSeek-OCR` cleanly on TPU v7x uncovered a concrete two-step tuning path:

### Step 1: Resolving the `TP=1` VMEM Overflow

* **Failure at `TP=1` (`--max-model-len 8192`):** During Pallas RPA attention kernel compilation (`RPAm-p_256-bq_1024_1024-bkv_2048_512`), XLA reported:
  ```text
  XLA:TPU compile permanent error: Used 79.89M of 63.94M vmem (Exceeded by 15.95M)
  ```
* **Resolution at `TP=2` (`--max-model-len 4096 --max-num-batched-tokens 2048`):** Halving attention heads per TPU core via `TP=2` (1 full TPU v7x chip) brought VMEM cleanly under the `63.94 MB` scratchpad limit. Server startup succeeded cleanly in `96.0–122.0 s` (warm cache) to `/health 200 OK`.

### Step 2: Avoiding `aten::_upsample_bicubic2d_aa.out` on Dynamic Multi-Crop Inputs

* **Failure on `1024x1400` multi-crop pages:** When an input image exceeds `640px` in width or height, `DeepSeek-OCR`'s dynamic tiling splits the page into `640x640` local crops (`tgt_size = 40` patches) alongside the `1024x1024` global view (`src_size = 64` patches). Inside `vllm/model_executor/models/deepencoder.py:337` (`get_abs_pos`), resizing the SAM absolute positional embedding from `64x64` to `40x40` calls:
  ```python
  F.interpolate(old_pos_embed, size=(tgt_size, tgt_size), mode="bicubic", antialias=True, align_corners=False)
  ```
  which raises on TorchTPU:
  ```text
  RuntimeError: operator 'aten::_upsample_bicubic2d_aa.out' is not implemented for TPU.
  ```
* **Working production pattern (`<= 640x640` single-view):** Passing document pages sized `<= 640x640` disables local sub-crops so DeepEncoder runs strictly on its native `1024x1024` global grid (`src_size == tgt_size == 64`), taking the `else: return abs_pos` branch and never invoking `aten::_upsample_bicubic2d_aa.out`. Every tested invoice and table page extracted at **100.0%** field accuracy using **288 vision tokens** per page (`6.56x` fewer visual tokens than `Qwen2.5-VL-7B`'s `1,889` tokens on `1024x1400`).

### Step 3: Warm vs Cold Vision-Encoder Batch Compilation

Because `DeepSeek-OCR` invokes its SAM+CLIP vision tower with `--mm-processor-cache-gb 0`, each newly observed vision batch size triggers a one-time ~11-second XLA compilation:

| Concurrency (`invoice_640x640`) | First Unwarmed Burst (`pages/min` / TTFT p95) | Pre-Warmed Steady State (`pages/min` / TTFT p95) |
|---:|---:|---:|
| `c=1` | 85.2 / 70.7 ms | **88.9** / 67.3 ms |
| `c=4` | 37.9 / 11,057.8 ms (includes compile) | **289.5** / 180.3 ms |
| `c=8` | 71.4 / 11,795.4 ms (includes compile) | **504.2** / 333.9 ms |
| `c=16` | 72.4 / 12,611.2 ms (includes compile) | **747.4** / 627.0 ms |

Production deployments should pre-warm target concurrency buckets (`c=1, 4, 8, 16`) during readiness probes before accepting live traffic.

---

## 2. `Qwen/Qwen2.5-VL-7B-Instruct` Quality & Tensor-Parallelism Notes

### Input Font Legibility Verification

Initial test runs in a minimal container lacking TrueType fonts (`fonts-dejavu-core`) fell back to PIL's 10px bitmap font, causing illegible invoice renders and premature 13-token terminations (`0%` field match). Those runs were discarded. Installing `fonts-dejavu-core` (`DejaVuSans.ttf` / `DejaVuSans-Bold.ttf`) immediately restored crisp document rendering and **100.0%** invoice field extraction (`426` structured Markdown tokens per page). On `table_1024x1400`, 5 of 6 tracked keywords (`83.3%`) matched across all `TP=2` runs because the long title string clipped `FY2026` at the right pixel margin.

### `TP=2` (1 Chip) vs `TP=4` (2 Chips)

* **Per-chip efficiency:** On `invoice_1024x1400` at `c=16`, `TP=2` achieves `359.8 pages/min` (`2,556.2 tok/s`) on 1 chip, whereas `TP=4` achieves `397.3 pages/min` (`2,821.2 tok/s`) across 2 chips (`198.7 pages/min/chip`). Running two independent `TP=2` replicas across 2 chips provides **`1.81x` higher aggregate throughput** than a single `TP=4` instance.
* **Stability:** `TP=2` maintained clean structured Markdown across every tested concurrency (`c=1, 4, 8, 16`) on invoices, tables, and high-resolution `1536x2048` pages (`145.1 pages/min` at `c=4`, `411` output tokens, `100.0%` field match). At `TP=4` (`1` KV head per shard on Qwen2.5-VL's 4 KV heads), occasional `addCriterion` decoding drift appeared on `table_1024x1400` (`c=1, 16`) and `highres_invoice_1536x2048` (`c=8`); full raw rows are preserved in `results/v7x_qwen25_vl_7b_ocr_tp4.jsonl` for transparency.

---

## 3. Dual-Engine Audit (`sglang-torchtpu v0.2.0`)

Both `Qwen/Qwen2.5-VL-7B-Instruct` and `deepseek-ai/DeepSeek-OCR` are registered in `sglang.srt.models.registry.ModelRegistry` and were tested live on TPU v7x (`TP=2`, `bfloat16`, `--device tpu --mem-fraction-static 0.70`) using `sglang-torchtpu v0.2.0`:

| Model | Engine | Config | Load / Startup | Outcome | Root Cause |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `sglang-torchtpu v0.2.0` | `TP=2` | Loaded weights (`161.0 s`) | Failed (`rc=-9`) | `sglang_tpu/mm_embed_tpu.py:139` (`RuntimeError: index_put_(): expected tensor to be on tpu, got cpu`) |
| `deepseek-ai/DeepSeek-OCR` | `sglang-torchtpu v0.2.0` | `TP=2` | Loaded weights (`67.0 s`) | Failed (`rc=-9`) | `sglang_tpu/mm_embed_tpu.py:139` (`RuntimeError: index_put_(): expected tensor to be on tpu, got cpu`) |

### Upstream Root Cause (`sglang_tpu/mm_embed_tpu.py`)

Inside `sglang_tpu/mm_embed_tpu.py`, `_HostRowBuffer.__init__` allocates `self.matrix = torch.zeros((rows, hidden), dtype=dtype)` without `device="cpu"` while the active default device is TPU. When `_place_mm_rows` scatters CPU host vision embeddings (`embedding.to_host()`) into `self.matrix` at line 139 (`self.matrix[indices] = rows`), PyTorch raises:

```text
File ".../sglang_tpu/mm_embed_tpu.py", line 247, in embed_mm_on_host
File ".../sglang_tpu/mm_embed_tpu.py", line 205, in _place_mm_rows
  buffer.write(indices, embedding.to(dtype))
File ".../sglang_tpu/mm_embed_tpu.py", line 139, in write
  self.matrix[indices] = rows
RuntimeError: index_put_(): expected tensor to be on tpu, got cpu
```

Until that host buffer allocation explicitly specifies `device="cpu"`, `vllm-torchtpu v0.33.0` is the verified serving engine for both OCR models on TPU v7x.

---

## 4. Raw Result Files

* [`results/v7x_deepseek_ocr_tp2_singleview.jsonl`](results/v7x_deepseek_ocr_tp2_singleview.jsonl) — Warm steady-state `DeepSeek-OCR` `TP=2` measurements (`invoice_640x640_singleview` and `table_640x640_singleview`, `c=1..16`).
* [`results/v7x_qwen25_vl_7b_ocr_tp2.jsonl`](results/v7x_qwen25_vl_7b_ocr_tp2.jsonl) — `Qwen2.5-VL-7B-Instruct` `TP=2` crisp TrueType measurements (`invoice_1024x1400`, `table_1024x1400`, `highres_invoice_1536x2048`).
* [`results/v7x_qwen25_vl_7b_ocr_tp4.jsonl`](results/v7x_qwen25_vl_7b_ocr_tp4.jsonl) — `Qwen2.5-VL-7B-Instruct` `TP=4` scaling comparison.
* [`results/v7x_sglang_dual_engine_audit.jsonl`](results/v7x_sglang_dual_engine_audit.jsonl) — Live `sglang-torchtpu v0.2.0` startup/failure records for `Qwen2.5-VL-7B-Instruct` and `DeepSeek-OCR`.

