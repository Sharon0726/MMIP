---
base_model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
library_name: peft
tags: [dpo, lora]
---

# Week6 視覺 DPO adapter

目標：列出圖片中存在的幾何形狀，避免多說不存在的種類。2026-11-04在Colab T4實際訓練，2026-11-09補齊模型說明。

- 基底：[HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)，固定revision `482adb537c021c86670beed01cd58990d01e72e4`。
- 資料：120組規則合成偏好，20組驗證；測試資料未用於更新或挑checkpoint。
- 4-bit NF4基底＋LoRA，r=8、alpha=16、dropout=0.05、q_proj/v_proj；DPO beta=0.1，2 epochs、60步。
- [訓練與評估報告](../../REPORT.md)、[完整可執行程式](../../MMIP_Week6_Colab.ipynb)。
- 這是adapter，不是完整模型。需先在Colab以相同revision和4-bit設定載入基底，再使用 `PeftModel.from_pretrained(base_model, adapter_directory)`；tokenizer／processor從本資料夾載入。完整提示與生成設定應沿用Notebook。
- 基底與tokenizer來源依上游[LICENSE](LICENSE)；不將上游著作宣稱為本作業原創。

單一seed與小型合成資料，不適合推論全面模型能力；文字測試有天花板效應，視覺微調改善有限。權重保留實際測試的版本，不因成績不佳改報結果。
