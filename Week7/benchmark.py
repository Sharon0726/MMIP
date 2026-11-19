"""Run A/B/C/D on the same held-out cases, with measured latency and compute cost."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from core import detect_attack, neutralize_instructions, parse_route, safe_fallback
from models import TextRouter, VisionCaptioner

ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


class FastRouter:
    """Experiment D: small trained decision model, with a conservative abstain path."""

    def __init__(self, train_path: Path):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        rows = read_jsonl(train_path)
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20261117),
        )
        texts = [r["turns"][-1] + " " + r.get("image_note", "") for r in rows]
        labels = [r["target"]["route"] for r in rows]
        self.model.fit(texts, labels)

    def __call__(self, turns: list[str], image_note: str) -> dict:
        text = neutralize_instructions(turns[-1] if turns else "")
        note = neutralize_instructions(image_note)
        probs = self.model.predict_proba([text + " " + note])[0]
        index = int(probs.argmax())
        confidence = float(probs[index])
        route = str(self.model.classes_[index])
        if confidence < 0.25:
            route = "clarify"
        return {
            "route": route,
            "intent": {
                "technical_support": "故障排查", "warranty": "保固維修",
                "order": "查詢訂單", "return_refund": "退貨退款",
                "sales": "產品諮詢", "human": "轉人工",
                "clarify": "確認需求", "out_of_scope": "非產品問題",
            }[route],
            "confidence": round(confidence, 4),
            "needs_clarification": route == "clarify",
            "summary": text[:60],
            "image_evidence": note[:60],
            "safety_flag": detect_attack((turns[-1] if turns else "") + " " + image_note),
        }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    return sorted_values[min(len(values) - 1, round((len(values) - 1) * p))]


def evaluate(vision: str, gpu_hourly_usd: float, output: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark A/B/C 與圖片理解必須在 Colab 或遠端 CUDA GPU 上執行")
    if not (ROOT / "artifacts" / "router_lora" / "adapter_config.json").exists():
        raise FileNotFoundError("請先在遠端 GPU 執行 train.py，產生 B 方法的 LoRA adapter")
    benchmark = read_jsonl(ROOT / "data" / "benchmark.jsonl")
    train_path = ROOT / "data" / "train.jsonl"
    fast = FastRouter(train_path)
    captioner = VisionCaptioner() if vision == "model" else None
    notes = {}
    vision_times = {}
    for case in benchmark:
        if not case["image"]:
            notes[case["id"]] = ""
            vision_times[case["id"]] = 0.0
            continue
        if vision == "oracle":
            notes[case["id"]] = case["expected_image_fact"]
            vision_times[case["id"]] = 0.0
        else:
            start = time.perf_counter()
            notes[case["id"]] = captioner(str(ROOT / case["image"]))
            vision_times[case["id"]] = time.perf_counter() - start
    router = TextRouter(adapter_path=str(ROOT / "artifacts" / "router_lora"))
    rows = []
    for case in benchmark:
        for method in "ABCD":
            note = notes[case["id"]]
            start = time.perf_counter()
            try:
                raw = fast(case["turns"], note) if method == "D" else router(method, case["turns"], note)
                parsed, error = parse_route(raw)
            except Exception as exc:
                raw, parsed, error = "", None, type(exc).__name__ + ": " + str(exc)
            seconds = time.perf_counter() - start
            if parsed:
                parsed["safety_flag"] = bool(parsed["safety_flag"] or detect_attack(" ".join(case["turns"]) + note))
            total_seconds = seconds + vision_times[case["id"]]
            rows.append({
                "id": case["id"], "method": method, "category": case["category"],
                "expected_route": case["expected_route"],
                "predicted_route": parsed["route"] if parsed else "",
                "format_valid": int(parsed is not None),
                "route_correct": int(parsed is not None and parsed["route"] == case["expected_route"]),
                "safety_flag": int(parsed["safety_flag"]) if parsed else 0,
                "router_seconds": round(seconds, 5),
                "vision_seconds": round(vision_times[case["id"]], 5),
                "total_seconds": round(total_seconds, 5),
                "gpu_cost_usd": round(
                    (vision_times[case["id"]] + (seconds if method != "D" else 0))
                    * gpu_hourly_usd / 3600, 8
                ),
                "image_note": note,
                "raw": json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else raw,
                "error": error or "",
            })
            print(
                f"{case['id']} {method}: route={rows[-1]['predicted_route'] or 'INVALID'} "
                f"valid={rows[-1]['format_valid']} correct={rows[-1]['route_correct']} "
                f"{total_seconds:.2f}s {error or ''}", flush=True,
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "all_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in "ABCD":
        subset = [r for r in rows if r["method"] == method]
        latencies = [r["total_seconds"] for r in subset]
        summary[method] = {
            "n": len(subset),
            "format_valid_rate": round(statistics.mean(r["format_valid"] for r in subset), 3),
            "route_accuracy": round(statistics.mean(r["route_correct"] for r in subset), 3),
            "median_seconds": round(statistics.median(latencies), 3),
            "p95_seconds": round(percentile(latencies, .95), 3),
            "total_gpu_cost_usd": round(sum(r["gpu_cost_usd"] for r in subset), 5),
        }
    metadata = {
        "vision": vision,
        "gpu_hourly_usd_assumption": gpu_hourly_usd,
        "hardware": torch.cuda.get_device_name(0),
        "note": "成本為 GPU 推論時間乘指定費率的估算；D 的 CPU 推論不計 GPU 費用。非 Colab 實際帳單；模型下載與訓練另計。",
        "summary": summary,
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Week7 實測結果", "",
        f"圖片來源：{'Qwen2.5-VL-3B-Instruct 實際觀察' if vision == 'model' else '人工標註（僅測分流節點）'}",
        f"GPU：{metadata['hardware']}；估算費率：USD {gpu_hourly_usd}/小時", "",
        "| 方法 | 筆數 | 結構正確率 | 分流正確率 | 延遲中位數 | P95 | GPU 等值成本 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in summary.items():
        lines.append(
            f"| {method} | {item['n']} | {item['format_valid_rate']:.1%} | "
            f"{item['route_accuracy']:.1%} | {item['median_seconds']:.3f}s | "
            f"{item['p95_seconds']:.3f}s | ${item['total_gpu_cost_usd']:.5f} |"
        )
    lines.extend(["", metadata["note"], "",
                  "A = Prompt；B = SFT LoRA；C = 逐 token JSON Schema 約束；D = 字元 n-gram + 邏輯回退。",
                  "四種方法使用相同模型可見的對話和圖片觀察；D 是較小的 System One 節點。",
                  "結構正確率要求全部欄位、型別與 route enum 正確。分流正確率按預先標註的 route 判斷。",
                  "完整逐題輸出與錯誤原因在 all_results.csv。"])
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", choices=["model", "oracle"], default="model")
    parser.add_argument("--gpu-hourly-usd", type=float, default=0.0)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "benchmark_model"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.vision, args.gpu_hourly_usd, Path(args.output)),
                     ensure_ascii=False, indent=2))
