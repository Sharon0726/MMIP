"""Replay measured attack outputs through the UI controller without GPU inference."""

import argparse
import csv
import json
from pathlib import Path

from core import ConversationController, ConversationState

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=ROOT / "artifacts/benchmark_model/all_results.csv")
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/attack_replay.json")
args = parser.parse_args()
benchmark = {
    row["id"]: row
    for row in map(json.loads, (ROOT / "data/benchmark.jsonl").read_text(encoding="utf-8").splitlines())
}
with args.results.open(encoding="utf-8-sig", newline="") as f:
    rows = [row for row in csv.DictReader(f) if row["id"] in {"B15", "B16"}]

results = []
for row in rows:
    case = benchmark[row["id"]]
    controller = ConversationController(
        lambda turns, note, raw=row["raw"]: raw,
        lambda path, note=row["image_note"]: note,
    )
    state = ConversationState()
    reply, route = controller.process(state, case["turns"][-1], case["image"])
    results.append({
        "id": row["id"],
        "method": row["method"],
        "raw_route": row["predicted_route"] or "invalid_schema",
        "guarded_route": route["route"],
        "safety_flag": route["safety_flag"],
        "expected_route": case["expected_route"],
        "guarded_correct": route["route"] == case["expected_route"],
    })

assert len(results) == 8
assert all(r["guarded_correct"] and r["safety_flag"] for r in results)
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
