"""Re-evaluate D using the exact VLM observations cached by an earlier full benchmark."""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from benchmark import FastRouter, ROOT, percentile, read_jsonl
from core import detect_attack, parse_route


def main() -> None:
    folder = ROOT / "artifacts" / "benchmark_model"
    csv_path = folder / "all_results.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    cases = {r["id"]: r for r in read_jsonl(ROOT / "data" / "benchmark.jsonl")}
    metadata = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    fast = FastRouter(ROOT / "data" / "train.jsonl")
    for row in rows:
        if row["method"] != "D":
            continue
        case = cases[row["id"]]
        note = row["image_note"]
        start = time.perf_counter()
        raw = fast(case["turns"], note)
        seconds = time.perf_counter() - start
        parsed, error = parse_route(raw)
        vision_seconds = float(row["vision_seconds"])
        row.update({
            "predicted_route": parsed["route"] if parsed else "",
            "format_valid": int(parsed is not None),
            "route_correct": int(parsed is not None and parsed["route"] == case["expected_route"]),
            "safety_flag": int(parsed["safety_flag"] or detect_attack(" ".join(case["turns"]) + note)) if parsed else 0,
            "router_seconds": round(seconds, 5),
            "total_seconds": round(seconds + vision_seconds, 5),
            "gpu_cost_usd": round(vision_seconds * metadata["gpu_hourly_usd_assumption"] / 3600, 8),
            "raw": json.dumps(raw, ensure_ascii=False),
            "error": error or "",
        })
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in "ABCD":
        subset = [r for r in rows if r["method"] == method]
        latencies = [float(r["total_seconds"]) for r in subset]
        metadata["summary"][method] = {
            "n": len(subset),
            "format_valid_rate": round(statistics.mean(int(r["format_valid"]) for r in subset), 3),
            "route_accuracy": round(statistics.mean(int(r["route_correct"]) for r in subset), 3),
            "median_seconds": round(statistics.median(latencies), 3),
            "p95_seconds": round(percentile(latencies, .95), 3),
            "total_gpu_cost_usd": round(sum(float(r["gpu_cost_usd"]) for r in subset), 5),
        }
    metadata["d_rerun"] = "D re-evaluated on the identical cached VLM observations after enforcing schema field lengths"
    (folder / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Week7 實測結果", "",
        "圖片來源：Qwen2.5-VL-3B-Instruct 實際觀察",
        f"GPU：{metadata['hardware']}；估算費率：USD {metadata['gpu_hourly_usd_assumption']}/小時", "",
        "| 方法 | 筆數 | 結構正確率 | 分流正確率 | 延遲中位數 | P95 | GPU 等值成本 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in metadata["summary"].items():
        lines.append(
            f"| {method} | {item['n']} | {item['format_valid_rate']:.1%} | "
            f"{item['route_accuracy']:.1%} | {item['median_seconds']:.3f}s | "
            f"{item['p95_seconds']:.3f}s | ${item['total_gpu_cost_usd']:.5f} |"
        )
    lines.extend(["", metadata["note"], "",
                  "D 修正輸出欄位長度後，以同一次實測快取的圖片觀察重新計時；A/B/C 保持原始輸出。",
                  "完整逐題輸出與錯誤原因在 all_results.csv。"])
    (folder / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
