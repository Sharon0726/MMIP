# MMIP Week7 執行說明

本資料夾依老師的三張 Week7 作業投影片完成多模態多輪客服、四種分流節點比較，以及聊天介面整合。先看 [老師要求逐項對照](RUBRIC.md)、[完整報告](REPORT.md)、[架構圖](architecture.svg)、[RTX 3090 實測表](artifacts/benchmark_model/results.md)與 [Colab T4 實測表](artifacts/colab_t4_20261119/benchmark_model/results.md)。兩次實驗各有 A/B/C/D 的 20 題真實模型結果及 B 的訓練權重。[真實模型瀏覽器端到端驗證](artifacts/colab_t4_20261119/ui_e2e.json)也已完成。上傳檔案的完整性可用 [SHA-256 清單](SHA256SUMS.json)核對。

講義原檔與作業投影片截圖未公開上傳。訓練與模型推論在遠端 RTX 3090 和 Google Colab T4 完成；本機只做程式、資料及假模型 UI 測試。

## 最容易的方式：Google Colab

1. 在 [Google Colab 直接開啟重跑筆記本](https://colab.research.google.com/github/Sharon0726/MMIP/blob/main/Week7/MMIP_Week7_Colab.ipynb)。
2. 選「執行階段 → 變更執行階段類型 → GPU」。
3. 第一個程式區塊會自行解開筆記本內含的作業檔，不需再上傳壓縮包。
4. 按順序執行安裝、資料驗證、LoRA 訓練、四方法 Benchmark、下載結果、啟動介面。安裝步驟會移除 Colab 目前與 PEFT 不相容的預裝 `torchao`；主要步驟失敗時會停止。
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
| `artifacts/colab_t4_20261119/` | T4 獨立訓練權重、80 筆逐題結果、攻擊重播及真實模型瀏覽器測試 |
| `MMIP_Week7_Executed_Colab_T4.ipynb` | Colab 實際執行記錄；保留首次依賴錯誤與修正後重跑輸出 |

**實測狀態**：RTX 3090 與 Colab T4 各完成 B 的 3 epoch SFT LoRA 訓練及 A/B/C/D 的 20 題評估。B 的分流正確率分別是 65% 與 55%；完整限制與分析見報告。八項控制流程測試、假模型 UI 回呼測試及 Colab 真實模型的瀏覽器端到端測試已通過；Colab 暫時介面於驗證後停止。
