# Week7 作業要求對照

依老師三張作業投影片逐項核對。下表列的是已保存、可查核的成果；不代表教師評分。

| 作業／配分 | 老師要求 | 對應成果與證據 |
|---|---|---|
| 一／5 | 設定客服業務情境、下游任務與結構輸出 Spec | `REPORT.md` 作業一第 1–2 節；`core.py` 的八種路由與 `ROUTE_SCHEMA` |
| 一／10 | 技術架構圖，包含完整角落情境 | `architecture.svg`、`REPORT.md` 作業一第 3–4 節 |
| 二 A／6 | 完全靠 Prompt 產生結構輸出 | `models.py` 的 `TextRouter("A")`；原始輸出在 `artifacts/benchmark_model/all_results.csv` |
| 二 B／8 | SFT 或 RLHF 訓練結構輸出 | `train.py` 的 SFT LoRA、160 筆 `data/train.jsonl`、`artifacts/router_lora/` 與 `artifacts/colab_t4_20261119/router_lora/` 的兩次訓練權重及紀錄 |
| 二 C／8 | Decoding 機制導引結構輸出 | `models.py` 的 `TextRouter("C")`，以 LM Format Enforcer 逐 token 約束 JSON Schema |
| 二 D／8 | System One 分流機制輸出結構資料 | `benchmark.py` 的 `FastRouter`，字元 n-gram 分類及低信心回退 |
| 二進階／10 | 至少 10 筆合理 Benchmark | `data/benchmark.jsonl` 有 20 筆獨立案例，含多輪、照片、換題、攻擊 |
| 二進階／10 | 比較 A–D 結構正確性、時間、成本 | `REPORT.md` 作業二第 3 節；`artifacts/benchmark_model/` 與 `artifacts/colab_t4_20261119/benchmark_model/` 各有 80 筆逐題結果與統計 |
| 三／5 | 有分流及多輪多模態能力的聊天機器人 | `app.py` 整合文字、圖片、對話狀態、路由與待確認工單；`verify.py` 八項控制流程測試及 `artifacts/colab_t4_20261119/ui_e2e.json` 的真實模型瀏覽器測試 |
| 三／5 | 合理的聊天 UI | `app.py` 的 Gradio 聊天區、圖片上傳、方法選擇、工單確認、新話題按鈕；`ui_smoke.py` 假模型回呼測試及 Colab T4 真實模型操作紀錄 |
| 三／5 | 優化一項 UX 並證明有效 | 選「完成一題後銜接下一話題」；`verify.py` 的 `test_topic_switch_drops_old_context` 驗證脈絡及主題 ID 重置 |
| 三進階／10 | 突然提出無脈絡問題，優化並驗證 | `core.py` 的 `out_of_scope` 處理；`verify.py` 測試；Colab 真實模型瀏覽器實測「樂透」不污染原訂單工單 |
| 三進階／10 | 文字或圖片攻擊，防護並驗證 | `core.py` 的攻擊訊號與指令中和；`verify.py` 控制測試；兩次 GPU 實測輸出的八筆離線重播；Colab 真實模型瀏覽器送出文字及圖片攻擊 |

## 實測範圍與限制

- A–D 的分流準確率是**分流節點原始輸出**，未套用聊天控制層的攻擊防護。RTX 3090 的 B 在 20 題答對 13 題；獨立 Colab T4 重跑答對 11 題。不能宣稱可靠處理真實客服。
- 兩道攻擊題的原始模型輸出都沒有正確分流；八筆離線重播經控制層後均改回正確部門。這證明已知攻擊樣式的控制流程，不能推論未知攻擊也有效。
- UX、題外問題及攻擊防護有程式測試、離線重播及 Colab 真實 GPU 模型的瀏覽器端到端操作證據；沒有真人可用性研究。第一次瀏覽器測試發現兩個分流缺陷，修正後覆測通過，過程保留在 `artifacts/colab_t4_20261119/ui_e2e.json`。
- 圖片與訓練資料為合成教學案例；成本依 USD 1／GPU 小時估算，不是實際帳單。
