# MMIP Week7 執行說明

本資料夾依老師的三張 Week7 作業投影片完成多模態多輪客服、四種分流節點比較，以及聊天介面整合。先看 [老師要求逐項對照](RUBRIC.md)、[完整報告](REPORT.md)、[架構圖](architecture.svg)與[實測表](artifacts/benchmark_model/results.md)。A/B/C/D 各有 20 題實測資料；B 的訓練權重也已保存。上傳檔案的完整性可用 [SHA-256 清單](SHA256SUMS.json)核對。

講義原檔與作業投影片截圖未公開上傳。訓練與模型推論在遠端 GPU 完成；本機只做程式、資料及假模型 UI 測試。

## 最容易的方式：Google Colab

1. 在 [Google Colab 直接開啟重跑筆記本](https://colab.research.google.com/github/Sharon0726/MMIP/blob/main/Week7/MMIP_Week7_Colab.ipynb)。
2. 選「執行階段 → 變更執行階段類型 → GPU」。
3. 第一個程式區塊會自行解開筆記本內含的作業檔，不需再上傳壓縮包。
4. 按順序執行安裝、資料驗證、LoRA 訓練、四方法 Benchmark、下載結果、啟動介面。
5. 重跑後的實測會出現在 `artifacts/benchmark_model/results.md`；逐題原始回覆在 `all_results.csv`。下載的 `MMIP_Week7_results.zip` 含 adapter 與實測資料。

Colab 執行階段可能中斷。訓練完成後建議先下載結果。合成資料不是實際售後客訴；報告應標明此限制。

## 共用 GPU 主機

本次已完成訓練與實測，相關程序已停止。共用主機有其他人的 GPU 工作時，不要在該主機啟動訓練、Benchmark 或聊天介面；可使用上面的 Colab 筆記本重跑。

## 檔案用途

| 檔案 | 用途 |
|---|---|
| `core.py` | 結構輸出規格、對話／換題狀態、安全回退、工單 |
| `make_data.py`、`data/` | 固定種子的 160 筆訓練例、20 題測試及五張合成圖片 |
| `models.py` | 共用 VLM 圖片理解、A/B/C 文字模型 |
| `train.py` | B 的 SFT LoRA 訓練 |
| `benchmark.py` | A/B/C/D 公平比較、計時、成本估算 |
| `app.py` | 多模態多輪聊天介面與待確認工單 |
| `verify.py` | 不需 GPU 的資料、換題、題外、注入和工單測試 |
| `ui_smoke.py` | 用假模型檢查 Gradio 介面與按鈕回呼，不用 GPU |
| `replay_attacks.py` | 用實測模型輸出離線重播文字與圖片攻擊 |
| `REPORT.md` | 三題完整論述、評量定義與限制 |
| `RUBRIC.md` | 老師三張作業圖逐項要求與證據對照 |
| `artifacts/router_lora/` | 已訓練的 B 方法 adapter 與訓練紀錄 |
| `artifacts/benchmark_model/` | 20 題 × 4 方法的原始結果與摘要 |
| `artifacts/attack_replay.json` | 八筆攻擊案例經聊天室控制層處理的結果 |
| `artifacts/ui_smoke.json` | 假模型 UI 回呼測試紀錄與限制 |

**實測狀態**：B 已完成 3 epoch SFT LoRA 訓練；A/B/C/D 各完成 20 題評估。B 的分流正確率最高，為 65%；完整限制與分析見報告。六項控制流程測試與假模型 UI 回呼測試通過；遠端 UI 已停止，尚未完成真實 GPU 模型的瀏覽器端到端驗證。
