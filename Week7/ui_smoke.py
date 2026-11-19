"""Check Gradio UI wiring with stub routers; never load or train GPU models."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import app
import gradio as gr
from core import ConversationState


def fake_route(*args):
    return {
        "route": "technical_support",
        "intent": "故障排查",
        "confidence": 0.8,
        "needs_clarification": False,
        "summary": "E03 故障，風扇不轉",
        "image_evidence": "螢幕顯示 E03",
        "safety_flag": False,
    }


with tempfile.TemporaryDirectory() as temp:
    with (
        patch.object(app, "ROOT", Path(temp)),
        patch.object(app, "FastRouter", lambda path: fake_route),
        patch.object(app, "TextRouter", lambda adapter_path: fake_route),
        patch.object(app, "VisionCaptioner", lambda: lambda path: "螢幕顯示 E03"),
    ):
        demo = app.create_app()
        callbacks = {dependency.fn.__name__: dependency.fn for dependency in demo.fns.values()}
        assert {"send", "confirm", "new_topic"} <= callbacks.keys()

        image = Path(__file__).resolve().parent / "data/images/error_e03.png"
        state = ConversationState()
        history, state, ticket, cleared_box = callbacks["send"](
            {"text": "E03 故障，風扇不轉", "files": [str(image)]},
            [], state, "B",
        )
        assert len(history) == 3
        assert state.pending_ticket and state.pending_ticket["route"] == "technical_support"
        assert "技術支援" in ticket

        state, history, ticket = callbacks["confirm"](state, history)
        assert state.pending_ticket is None
        outbox = Path(temp) / "artifacts/outbox.jsonl"
        assert outbox.exists() and len(outbox.read_text(encoding="utf-8").splitlines()) == 1

        new_state, new_history, message = callbacks["new_topic"]()
        assert not new_state.active_turns and not new_history and "新" in message

record = {
    "gradio_version": gr.__version__,
    "mode": "stub_models_no_gpu",
    "checks": [
        "components_created",
        "text_and_image_send_callback",
        "ticket_confirmation_callback",
        "new_topic_callback",
    ],
    "passed": True,
    "limitation": "Does not test real GPU models or browser interactions.",
}
out = Path(__file__).resolve().parent / "artifacts/ui_smoke.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print("UI 組件、文字加圖片送出、工單確認與新話題回呼均通過（假模型；無 GPU）。")
