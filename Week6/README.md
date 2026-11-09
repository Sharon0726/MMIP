# MMIP Week6：大型語言模型、視覺語言模型與偏好微調

**蘇芯儀 / Sharon0726** · 簡報日期 2026-10-21 · 實驗日期 2026-11-04 · 查核整理 2026-11-09

依 Week6 資料夾中老師「第七堂：多模態語言模型如何運作」簡報第 **64–66 頁**完成 A、B、C 基礎及進階實驗。簡報列出 **15 項、合計 90 分**；本表是證據對照，不是教師評分或滿分保證。講義原檔不重公開上傳。

## 先看哪裡

- [完整實驗報告與結果解讀](REPORT.md)（GitHub 可直接閱讀）
- [已執行 Notebook：程式與原始輸出](MMIP_Week6_Executed.ipynb)
- [在 Colab 開啟乾淨重跑入口](https://colab.research.google.com/github/Sharon0726/MMIP/blob/main/Week6/MMIP_Week6_Colab.ipynb)
- [原 Colab 實驗筆記本](https://colab.research.google.com/drive/1xSs_PHP1XT1a0cWR0tGHyqlFB6pNiOSo)（存取權由原帳號管理）
- [原始 HTML 報告](artifacts/report.html)（下載後開啟，圖表已內嵌）
- [離線查核結果](verification.json)與[查核程式](verify_results.py)

## 老師完整要求與對應證據

| 頁／要求 | 分數 | 本次實作 | 可查核證據 |
|---|---:|---|---|
| 64：部署並執行一個開源 LLM | 5 | Qwen2.5-1.5B-Instruct | [環境與模型版本](artifacts/environment.json)、[120份回答](artifacts/A_results.json) |
| 64：20道不同能力的自訂 Instruction Benchmark | 5 | 算術、邏輯、閱讀、翻譯、格式各4題 | [20題與標準答案](artifacts/A_benchmark.json) |
| 64：提示工程及量化指標或商用評估 | 5 | baseline / few-shot / CoT；獨立10題選提示 | [開發集](artifacts/A_development.json)、[選擇紀錄](artifacts/A_prompt_selection.json)、[成績](artifacts/A_summary.csv) |
| 64：預設 Prompt 下的模型量化，量化指標與商用評估 | 5 | FP16 / INT8 / NF4；相同baseline提示 | [逐題回答與商用評分](artifacts/A_judged.json)、[結果表](artifacts/A_summary.csv) |
| 64進階：預設精度推論的平均 GPU 記憶體 | 5 | FP16未量化基準；NVML時間積分平均 | [平均與峰值](artifacts/A_fresh_process_memory_summary.csv)、[FP16原始取樣](artifacts/memory_fp16_baseline.json) |
| 64進階：優化提示＋低精度，顯存及成效比較 | 5 | 開發集選few-shot，結合4-bit；與FP16比較 | [獨立程序測量](artifacts/A_fresh_process_memory_summary.csv)、[品質](artifacts/A_summary.csv) |
| 65：部署兩個開源 VLM | 10 | SmolVLM2-2.2B、Qwen2.5-VL-3B | [固定版本](artifacts/B_model_revisions.json)、[40份原回答](artifacts/B_results.json) |
| 65：自訂20道視覺理解題 | 5 | 辨識、計數、位置、OCR、描述各4題 | [題庫](artifacts/B_benchmark.json)、[全部圖片](artifacts/B_benchmark_contact_sheet.png) |
| 65：同樣 Prompt，量化指標與商用評估 | 5 | 兩模型相同英文題目；共同修正評分規則 | [最終逐題評分](artifacts/B_v2_judged.json)、[比較表](artifacts/B_final_summary.csv) |
| 66：自選優化面向，至少100組偏好資料 | 5 | 訂單JSON擷取，缺失欄位不可捏造；120組 | [chosen/rejected資料](artifacts/C_train_preferences.json) |
| 66：另外100組與目標無關的通用 Benchmark | 5 | 算術、邏輯、英文、科學、大小比較各20題 | [通用題庫](artifacts/C_general_benchmark_100.json) |
| 66：實際微調LLM，量化目標及通用變化 | 10 | 4-bit QLoRA＋DPO，2 epochs／60步；前後各130題 | [訓練紀錄](artifacts/C_training_log.json)、[權重](artifacts/C_text_adapter/)、[前後比較](artifacts/C_final_summary.csv) |
| 66進階：VLM優化目標，至少100組資料 | 10 | 避免說出不存在的幾何形狀；120組視覺偏好 | [資料](artifacts/V_train_preferences.json)、[訓練圖](artifacts/vlm_preference_images/) |
| 66進階：實際微調VLM及量化評估 | 5 | SmolVLM2 DPO，2 epochs／60步；獨立20圖 | [訓練紀錄](artifacts/V_training_log.json)、[權重](artifacts/V_adapter/)、[比較](artifacts/V_summary.csv) |
| 66進階：畫質、明暗變化與平均表現 | 5 | 同20張原圖、亮度0.35、32×32＋JPEG20；前後各60題 | [擾動示意](artifacts/V_perturbation_examples.png)、[逐題結果](artifacts/V_all_answers.csv) |

## 如何重現

**閱讀與檢查不需要顯卡。** 在此資料夾執行：

```bash
python verify_results.py
```

僅需 Python 3.9+ 標準函式庫，會核對資料數量／切分、圖片、客觀評分、商用評分快取、平均顯存取樣、訓練步數及實際 LoRA 權重，並更新 `verification.json`。這是保存證據的離線重算，沒有重新跑模型或重新請商用模型評分。

**完整訓練使用 Colab：** 點上方乾淨入口，選 T4 GPU，由上往下執行。每次全流程使用新的 `/content/mmip_week6_時間戳` 資料夾，避免誤用舊測試快取。需要網路、Colab GPU 配額及內建 Gemini 評分額度。套件安裝後若 Colab 要求重啟 Python，重啟後從頭執行。程式最後會提供成果 ZIP 下載。

2026-11-04 已執行版本使用 Python 3.13.16、PyTorch 2.11.0+cu130、Tesla T4；完整版本在 [environment.json](artifacts/environment.json)。已固定三個模型的 checkpoint SHA；套件列在 [requirements-colab.txt](requirements-colab.txt)。原實驗未記錄 NumPy、scikit-image 的版本，不能保證逐位元重現。2026-11-09 的乾淨入口補齊依存安裝與路徑隔離，經語法及資料查核，**沒有冒稱整理後又完整重訓一次**。

## 資料與權重如何閱讀

`artifacts/` 保存實驗輸出的原始 JSON、CSV、圖片、商用評分請求／回覆、訓練紀錄、tokenizer／processor與兩份 LoRA adapter；沒有大型基底模型或 optimizer checkpoints。JSON 中的 `/content/mmip_week6/...` 是原 Colab 路徑；本機閱讀時對應 `artifacts/...`，查核程式會轉換。

- A：使用 `A_summary.csv` 看品質，`A_fresh_process_memory_summary.csv` 看獨立程序顯存；`A_summary.csv` 中的整張GPU用量保留作原始紀錄，不能與程序用量混為一談。
- B：**最終採用 `B_final_*`／`B_v2_judged.json`**；`B_summary.csv`、`B_judged_v1.json` 與 `B_comparison.png` 是保留的修正前紀錄。
- C：最終採用 `C_final_*`；無效JSON與缺失欄位填值分開處理。
- V：`V_summary.csv`、`V_before_results.json`、`V_after_results.json`；觀察F1、召回率、幻覺率及截斷數。
- 權重載入方式：[文字 adapter](artifacts/C_text_adapter/README.md)、[視覺 adapter](artifacts/V_adapter/README.md)。

## 必須保留的限制

1. 原始checkpoint為BF16，受T4條件限制以FP16作未量化基準；沒有原生BF16顯存實測。
2. 商用評分者是 `google/gemini-3.5-flash-lite`，視覺題讀取經核對的場景標註與模型答案，**沒有直接接收圖片**。盲評不提供受測模型名稱。
3. 偏好資料是規則合成、非真人回饋。文字目標30題微調前已全對，不能聲稱有提升；通用61%→60%不足以證明嚴重遺忘。
4. 單一seed、小型測試、有限詞彙與生成長度，不代表全面模型排名，也不保證微調一定改善。

實驗與報告由 AI 協助撰寫、執行與查核；分數均來自保存的實測資料。模型與影像來源見 [來源說明](SOURCES.md)。
