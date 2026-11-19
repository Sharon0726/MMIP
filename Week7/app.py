"""Experiment 3: multimodal, multi-turn customer support UI with a reviewed ticket outbox."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import FastRouter
from core import ConversationController, ConversationState, submit_ticket
from models import TextRouter, VisionCaptioner

ROOT = Path(__file__).resolve().parent


def create_app(default_method: str = "B"):
    import gradio as gr

    fast = FastRouter(ROOT / "data" / "train.jsonl")
    text_router = TextRouter(adapter_path=str(ROOT / "artifacts" / "router_lora"))
    vision = VisionCaptioner()

    def dispatch(method: str):
        def route(turns: list[str], note: str):
            return fast(turns, note) if method == "D" else text_router(method, turns, note)
        return ConversationController(route, vision)

    def show_ticket(state: ConversationState) -> str:
        if not state.pending_ticket:
            return "目前沒有待確認的工單。"
        ticket = state.pending_ticket
        return (
            f"### 待確認工單 {ticket['ticket_id']}\n"
            f"部門：**{ticket['department']}**  \n"
            f"摘要：{ticket['summary']}  \n"
            f"圖片觀察：{ticket['image_evidence'] or '無'}  \n"
            "按下「確認並建立工單」後，資料會寫入此執行環境的 outbox.jsonl。"
        )

    def send(message, history, state, method):
        message = message or {}
        text = message.get("text", "")
        files = message.get("files") or []
        path = str(files[0]) if files else None
        if not text and not path:
            return history, state, show_ticket(state), gr.MultimodalTextbox(value=None)
        controller = dispatch(method)
        reply, route = controller.process(state, text, path)
        history = list(history or [])
        if text:
            history.append({"role": "user", "content": text})
        if path:
            history.append({"role": "user", "content": gr.Image(value=path)})
        history.append({"role": "assistant", "content": reply})
        return history, state, show_ticket(state), gr.MultimodalTextbox(value=None)

    def confirm(state, history):
        try:
            ticket = submit_ticket(state, ROOT / "artifacts" / "outbox.jsonl")
            response = f"已建立工單 {ticket['ticket_id']}，交由{ticket['department']}待人工處理。"
        except ValueError:
            response = "目前沒有可送出的工單。"
        history = list(history or []) + [{"role": "assistant", "content": response}]
        return state, history, show_ticket(state)

    def new_topic():
        return ConversationState(), [], "已開啟新的話題。"

    with gr.Blocks(title="AirNest 多模態客服") as demo:
        gr.Markdown("# AirNest 多模態客服\n輸入問題、上傳產品照片，系統會先確認需求，再建立待人工處理的工單。")
        with gr.Row():
            method = gr.Radio(
                choices=["A", "B", "C", "D"], value=default_method,
                label="分流方法（A Prompt／B SFT／C 約束解碼／D System One）",
            )
            reset = gr.Button("開啟新話題")
        chat = gr.Chatbot(label="對話", type="messages", height=450)
        state = gr.State(ConversationState())
        box = gr.MultimodalTextbox(
            label="描述問題或上傳照片", file_types=["image"],
            placeholder="例如：我的清淨機顯示 E03，風扇不轉了",
            show_label=True,
        )
        send_button = gr.Button("傳送", variant="primary")
        ticket_view = gr.Markdown("目前沒有待確認的工單。")
        confirm_button = gr.Button("確認並建立工單")
        send_button.click(send, [box, chat, state, method], [chat, state, ticket_view, box])
        box.submit(send, [box, chat, state, method], [chat, state, ticket_view, box])
        confirm_button.click(confirm, [state, chat], [state, chat, ticket_view])
        reset.click(new_topic, outputs=[state, chat, ticket_view])
        gr.Markdown(
            "照片與訊息只用於目前對話及待確認工單。圖片中的操作指令會被視為圖片內容；"
            "此示範不連接正式訂單或保固系統。"
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Use only on Colab for a temporary public demo URL")
    args = parser.parse_args()
    create_app().launch(server_name=args.host, server_port=args.port, share=args.share)
