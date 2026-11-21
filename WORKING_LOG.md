# Working Log: Document OCR & Visual Parsing Bring-Up on TPU v7x

Chronological engineering log of license verification, `vllm-torchtpu v0.33.0` bring-up and tuning, `sglang-torchtpu v0.2.0` dual-engine auditing, and quality verification on Google Cloud TPU v7x.

## 1. License Verification & Test Document Generation

- Verified licenses before downloading weights: `deepseek-ai/DeepSeek-OCR` (MIT) and `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0).
- Initial OCR renders inside a minimal container lacking TrueType fonts (`fonts-dejavu-core`) fell back to PIL's 10px bitmap font and caused unreadable text (`0%` field match). Installed `fonts-dejavu-core` (`DejaVuSans.ttf` / `DejaVuSans-Bold.ttf`), restoring crisp invoice and table renders and **100.0%** field accuracy.

## 2. `Qwen/Qwen2.5-VL-7B-Instruct` (`vllm-torchtpu v0.33.0`)

- Served cleanly on 1 TPU v7x chip (`TP=2`, `--max-model-len 8192 --max-num-batched-tokens 8192`).
- Measured steady-state streaming HTTP throughput across `1024x1400` invoices (`359.8 pages/min`, `2,556.2 tok/s`, `25.4 ms` `c=1` TTFT, `100.0%` field match), dense financial tables (`404.0 pages/min`), and `1536x2048` high-res invoices (`145.1 pages/min`, `100.0%` field match).
- Evaluated `TP=4` (2 chips) scaling comparison: `397.3 pages/min` total (`198.7 pages/min/chip`), showing `1.81x` higher throughput per chip for independent `TP=2` replicas.

## 3. `deepseek-ai/DeepSeek-OCR` (`vllm-torchtpu v0.33.0`)

- **Failure 1 (`TP=1` VMEM overflow):** Pallas RPA attention exceeded the `63.94 MB` TPU v7x VMEM budget (`79.89 MB`). Moving to `TP=2` (`--max-model-len 4096 --max-num-batched-tokens 2048`) resolved VMEM cleanly and booted in `96–122 s`.
- **Failure 2 (Multi-crop `aten::_upsample_bicubic2d_aa.out`):** Pages larger than `640px` triggered dynamic local cropping inside `deepencoder.py:337`, calling antialiased bicubic interpolation (`aten::_upsample_bicubic2d_aa.out`, unimplemented on TPU).
- **Resolution:** Sizing inputs to single-view `640x640` kept DeepEncoder on its native `1024x1024` global grid (`288` visual tokens/page) with zero bicubic interpolation.
- **Warmup finding:** Vision-encoder batch sizes incur a one-time ~11 s compilation when `--mm-processor-cache-gb 0` is set. Pre-warming `c=1, 4, 8, 16` unlocked **747.4 pages/min** (`2,614.4 tok/s`, `100.0%` field match) on invoices and **655.4 pages/min** (`2,840.1 tok/s`, `100.0%` field match) on financial tables.

## 4. Dual-Engine Audit (`sglang-torchtpu v0.2.0`)

- Tested live startup of `Qwen/Qwen2.5-VL-7B-Instruct` (`TP=2`) and `deepseek-ai/DeepSeek-OCR` (`TP=2`) inside `sglang-torchtpu v0.2.0`.
- Both models loaded weights on TPU v7x and then crashed with the identical device-placement bug at `sglang_tpu/mm_embed_tpu.py:139` (`RuntimeError: index_put_(): expected tensor to be on tpu, got cpu`), caused by `_HostRowBuffer.__init__` calling `torch.zeros(...)` without `device="cpu"`.
- Recorded verbatim audit traces in `results/v7x_sglang_dual_engine_audit.jsonl` and `APPENDIX.md`.
