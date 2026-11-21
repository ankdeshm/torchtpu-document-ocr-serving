#!/usr/bin/env python3
"""Reproducible Document OCR & Structured Visual Parsing Benchmark on Cloud TPU v7x.

Renders crisp TrueType (DejaVuSans) enterprise document pages (invoices, multi-column
financial tables, and high-resolution pages) and benchmarks OpenAI-compatible vLLM
vision-language servers (`/v1/chat/completions`) with streaming TTFT, page throughput
(`pages/min`), output token rate, and automatic field-level OCR accuracy checks.

Requires `fonts-dejavu-core` (or equivalent TrueType font) so rendered text matches
production scanned/PDF document legibility.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError("No TrueType font found. Install `fonts-dejavu-core` first.")


def make_invoice(width: int, height: int, out_path: Path) -> Path:
    scale = width / 1024.0
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    ft = load_font(max(18, int(36 * scale)), bold=True)
    fh = load_font(max(14, int(24 * scale)), bold=True)
    fb = load_font(max(14, int(21 * scale)), bold=False)

    pad = int(60 * scale)
    y = int(50 * scale)
    d.text((pad, y), "CLOUD INFRASTRUCTURE INVOICE", fill="black", font=ft)
    y += int(60 * scale)
    d.line([(pad, y), (width - pad, y)], fill="black", width=max(2, int(3 * scale)))
    y += int(22 * scale)
    for k, v in [
        ("Invoice Number:", "INV-2026-8841"),
        ("Invoice Date:", "2026-11-20"),
        ("Payment Terms:", "Net 30 Days"),
        ("Bill To:", "Apex AI Systems Inc., San Francisco, CA"),
    ]:
        d.text((pad, y), k, fill="#222222", font=fh)
        d.text((pad + int(230 * scale), y), v, fill="black", font=fb)
        y += int(36 * scale)

    y += int(18 * scale)
    d.rectangle([pad, y, width - pad, y + int(44 * scale)], fill="#eeeeee", outline="black", width=2)
    cols = [pad + int(15 * scale), pad + int(450 * scale), pad + int(680 * scale)]
    for cx, h in zip(cols, ["Service Description", "Quantity", "Line Total"]):
        d.text((cx, y + int(9 * scale)), h, fill="black", font=fh)
    y += int(54 * scale)
    for desc, qty, tot in [
        ("Cloud Infrastructure Compute Pool", "160 hrs", "$48,000.00"),
        ("High-Speed Distributed Object Storage", "250 TB", "$10,500.00"),
        ("Enterprise Network Egress Bandwidth", "45 TB", "$3,600.00"),
        ("Managed Kubernetes Control Plane", "12 nodes", "$15,000.00"),
        ("Vector Search Index Hosting Tier", "8 shards", "$19,200.00"),
        ("24x7 Priority Engineering Support", "1 month", "$8,500.00"),
    ]:
        d.text((cols[0], y), desc, fill="black", font=fb)
        d.text((cols[1], y), qty, fill="black", font=fb)
        d.text((cols[2], y), tot, fill="black", font=fb)
        y += int(42 * scale)
        d.line([(pad, y - int(6 * scale)), (width - pad, y - int(6 * scale))], fill="#cccccc", width=1)

    y += int(24 * scale)
    for lbl, val in [("Subtotal:", "$104,800.00"), ("Tax (7.3%):", "$7,650.00"), ("Total Due:", "$112,450.00")]:
        f = fh if "Total" in lbl else fb
        d.text((cols[1], y), lbl, fill="black", font=f)
        d.text((cols[2], y), val, fill="black", font=f)
        y += int(40 * scale)

    img.save(out_path)
    return out_path


def make_table(width: int, height: int, out_path: Path) -> Path:
    scale = width / 1024.0
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    ft = load_font(max(18, int(30 * scale)), bold=True)
    fh = load_font(max(14, int(20 * scale)), bold=True)
    fb = load_font(max(14, int(19 * scale)), bold=False)

    pad = int(48 * scale)
    y = int(44 * scale)
    d.text((pad, y), "QUARTERLY PERFORMANCE REPORT FY2026", fill="black", font=ft)
    y += int(58 * scale)
    cols = [int(c * scale) for c in [55, 360, 510, 660, 810]]
    d.rectangle([pad, y, width - pad, y + int(42 * scale)], fill="#e8eef5", outline="black", width=2)
    for cx, h in zip(cols, ["Workload", "Q1 Ops/s", "Q2 Ops/s", "Q4 Ops/s", "Uptime"]):
        d.text((cx, y + int(9 * scale)), h, fill="black", font=fh)
    y += int(52 * scale)
    for r in [
        ("Video Diffusion", "142.5", "186.2", "294.0", "99.8%"),
        ("Image Generation", "610.4", "745.0", "1048.6", "99.9%"),
        ("Speech ASR", "1240.0", "1580.5", "2310.4", "99.9%"),
        ("Voice Synthesis", "480.2", "535.9", "704.3", "99.7%"),
        ("Dense Embeddings", "8420.0", "9610.0", "13480.0", "99.9%"),
        ("Cross-Encoder", "3120.5", "3690.2", "4895.0", "99.8%"),
        ("Document OCR", "285.4", "342.1", "512.6", "99.8%"),
        ("Robotics VLA", "188.0", "215.4", "296.5", "99.9%"),
    ]:
        for cx, val in zip(cols, r):
            d.text((cx, y), val, fill="black", font=fb)
        y += int(46 * scale)
        d.line([(pad, y - int(8 * scale)), (width - pad, y - int(8 * scale))], fill="#cccccc", width=1)

    img.save(out_path)
    return out_path


def call_ocr(base_url: str, model: str, b64_img: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    in_tok = out_tok = 0
    with urllib.request.urlopen(req, timeout=240) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                in_tok = obj["usage"].get("prompt_tokens", in_tok)
                out_tok = obj["usage"].get("completion_tokens", out_tok)
            choices = obj.get("choices") or []
            if choices:
                txt = (choices[0].get("delta") or {}).get("content") or ""
                if txt:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    chunks.append(txt)
    e2e = time.perf_counter() - t0
    text = "".join(chunks)
    if out_tok == 0 and text:
        out_tok = max(1, len(text.split()))
    return {
        "ttft_ms": ttft if ttft is not None else e2e * 1000.0,
        "e2e_s": e2e,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "text": text,
    }


def keyword_match_pct(text: str, keywords: list[str]) -> float:
    norm = text.lower().replace(",", "")
    hits = sum(1 for k in keywords if k.lower().replace(",", "") in norm)
    return round(100.0 * hits / max(1, len(keywords)), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark vision-language document OCR on TPU v7x.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="Served model name")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--profile-set", choices=["full", "singleview-640"], default="full")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work_dir = Path("/tmp/tpu_ocr_bench_pages")
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.profile_set == "singleview-640":
        pages = [
            (
                "invoice_640x640_singleview",
                make_invoice(640, 640, work_dir / "inv_640.png"),
                [1, 4, 8, 16],
                ["INV-2026-8841", "2026-11-20", "48000", "112450", "Net 30"],
            ),
            (
                "table_640x640_singleview",
                make_table(640, 640, work_dir / "tbl_640.png"),
                [1, 4, 8, 16],
                ["FY2026", "Video Diffusion", "Dense Embeddings", "Document OCR", "99.8%"],
            ),
        ]
        prompt = "Free OCR. Extract all text and tables into clean Markdown."
    else:
        pages = [
            (
                "invoice_1024x1400",
                make_invoice(1024, 1400, work_dir / "inv_1024.png"),
                [1, 4, 8, 16],
                ["INV-2026-8841", "2026-11-20", "Cloud Infrastructure", "48000", "112450", "Net 30"],
            ),
            (
                "table_1024x1400",
                make_table(1024, 1400, work_dir / "tbl_1024.png"),
                [1, 4, 8, 16],
                ["FY2026", "Video Diffusion", "Dense Embeddings", "Document OCR", "99.8%", "13480"],
            ),
            (
                "highres_invoice_1536x2048",
                make_invoice(1536, 2048, work_dir / "inv_1536.png"),
                [1, 4, 8],
                ["INV-2026-8841", "2026-11-20", "Cloud Infrastructure", "48000", "112450", "Net 30"],
            ),
        ]
        prompt = "Transcribe all text, numbers, and tables in this document image accurately into clean Markdown."

    out_path = Path(args.out) if args.out else None
    for name, img_path, cs, kws in pages:
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        for c in cs:
            # Warm this exact concurrency level so vision batch shapes are pre-compiled
            with ThreadPoolExecutor(max_workers=c) as ex:
                _ = list(ex.map(lambda _: call_ocr(args.base_url, args.model, b64, prompt, args.max_tokens), range(c)))
            n_reqs = max(4, c * 2) if c <= 8 else 16
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                res = list(ex.map(lambda _: call_ocr(args.base_url, args.model, b64, prompt, args.max_tokens), range(n_reqs)))
            wall = time.perf_counter() - t0
            ttfts = sorted(r["ttft_ms"] for r in res)
            e2es = sorted(r["e2e_s"] for r in res)
            rec = {
                "category": "ocr",
                "model": args.model,
                "engine": "vllm-torchtpu-0.33.0",
                "chip": "v7x",
                "tp": args.tp,
                "page_profile": name,
                "c": c,
                "n_pages": n_reqs,
                "wall_s": round(wall, 3),
                "avg_in_toks": round(sum(r["in_tok"] for r in res) / len(res), 1),
                "avg_out_toks": round(sum(r["out_tok"] for r in res) / len(res), 1),
                "pages_per_sec": round(n_reqs / wall, 2),
                "pages_per_min": round(60.0 * n_reqs / wall, 1),
                "out_tok_s": round(sum(r["out_tok"] for r in res) / wall, 2),
                "ttft_p50_ms": round(ttfts[len(ttfts) // 2], 2),
                "ttft_p95_ms": round(ttfts[int(0.95 * (len(ttfts) - 1))], 2),
                "page_e2e_p50_s": round(e2es[len(e2es) // 2], 3),
                "ocr_keyword_match_pct": keyword_match_pct(res[0]["text"], kws),
                "status": "ok",
            }
            print(json.dumps(rec), flush=True)
            if out_path:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
