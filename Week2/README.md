# HW2：MMIP 三題小作業

對應 `MMIP_第二堂_20260916_人工智慧、機器學習、深度學習_V4` 講義第 72–75 頁。
三題的基礎與進階要求都放在對應 Notebook，已附實際執行的文字、表格與圖片輸出。

## 1. 三份作業

| 作業 | Notebook | 內容 |
|---|---|---|
| Quiz 1 | [Quiz1_ML_Classification.ipynb](Quiz1_ML_Classification.ipynb) | 8 特徵模擬二元資料、縮放、Logistic Regression／Random Forest、門檻調整、混淆矩陣、四項指標與錯誤分析 |
| Quiz 2 | [Quiz2_Credit_Default_MLP.ipynb](Quiz2_Credit_Default_MLP.ipynb) | 指定信用卡資料、PyTorch MLP、原始／改良各 60 epochs、Dropout／L2、loss 曲線、指標比較與單筆預測 |
| Quiz 3 | [Quiz3_ROC_AUC_Comparison.ipynb](Quiz3_ROC_AUC_Comparison.ipynb) | 沿用 Quiz 2 的切分及 Improved MLP 機率、Random Forest、單模型與同圖 ROC／AUC、區分能力比較 |

每個程式 cell 都附用途，標示為「主目標」、「子目標」或「測試」。
完整結果與觀察整理在 [REPORT.md](REPORT.md)。

## 2. 檔案安排

| 路徑 | 用途 |
|---|---|
| `data/UCI_Credit_Card.csv` | 老師指定 Kaggle 的原始資料，已附在壓縮檔 |
| `data/SOURCE.md` | 資料來源、授權、SHA-256 與引用 |
| `artifacts/quiz2/credit_split.npz` | 同一組 train／validation 索引、客戶 ID、標籤、縮放資料及 scaler 統計量 |
| `artifacts/quiz2/mlp_validation_predictions.npz` | 兩個 MLP 的驗證機率，供 Quiz 3 直接讀取 |
| `artifacts/quiz2/*.pt` | 完整訓練後選出的最佳 MLP checkpoints |
| `artifacts/quiz2/metadata.json` | 超參數、最佳 epoch、套件版本與資料識別資訊 |
| `results/quiz1/`、`results/quiz2/`、`results/quiz3/` | CSV 指標、PNG 圖表與實测觀察 |
| `requirements.txt` | 執行環境套件 |

## 3. 本機執行

1. 解壓縮，保持完整 `HW2` 資料夾，不要只拿走其中一個 Notebook。
2. 使用 Python 3.11 或 3.12。這次已在 Python 3.12、PyTorch 2.6.0 CPU 執行。
3. 在終端機切到 `HW2` 目錄，安裝套件並啟動 Jupyter。

**子目標：安裝 CPU 版 PyTorch，再安裝其餘套件與啟動 Notebook。**

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m notebook
```

4. 依序開啟 Quiz 1 → Quiz 2 → Quiz 3，各自選擇「Restart Kernel and Run All」。VS Code 可使用「Run All」。
5. Quiz 2 會重新完整訓練兩個模型，並覆寫对应結果。重跑 Quiz 2 後也要重跑 Quiz 3。

Quiz 1 可獨立執行。Quiz 3 已附所需 artifacts，因此初次查看可直接執行；它不會重新切分資料。
在 GitHub 查看 `.ipynb` 可看到已保存的輸出，查看 `REPORT.md` 可直接閱讀結果圖表。
不同套件版本或硬體可能有微小數值差異；固定資料、seed 與模型設定的紀錄均已保存。

## 4. Colab 執行（可選）

1. 將 `HW2.zip` 上傳到 Colab 的 `/content`，並解壓。

**子目標：保留完整資料夾，指定 Notebook 的根目錄。**

```python
import zipfile
import os
with zipfile.ZipFile('/content/HW2.zip') as z:
    z.extractall('/content')
os.environ['HW2_ROOT'] = '/content/HW2'
```

2. 在 Colab 開啟對應 Notebook，於最前面補上 `os.environ['HW2_ROOT'] = '/content/HW2'` 的設定後由上到下執行。
3. Quiz 2 重跑後若換到新 Colab session，需保留更新後的整個 `HW2`，讓 Quiz 3 讀到最新 artifacts。

## 5. 放到 MMIP GitHub

1. 將解壓後的整個 `HW2` 資料夾，複製到你電腦上的 `MMIP` repository 根目錄。
2. 在 **MMIP 根目錄**開啟終端機，確認新檔，再提交與推送。

**主目標：把三題小作業放入既有 MMIP repository 的 HW2 目錄。**

```bash
git status
git add HW2
git commit -m "Add HW2 classification, credit default MLP and ROC evaluation"
git push
```

不需要重新建立 repository，也不需要把 zip 本身推上去。
此壓縮檔未對遠端 GitHub 做任何推送。

## 6. 方法與閱讀注意事項

- Quiz 1 的資料為人工模擬；Quiz 2、3 使用老師指定 Kaggle CSV，沒有用模擬資料替代。
- Quiz 1 以訓練集內三折交叉驗證挑選 F1 門檻，最後才評估驗證集；另明確比較 0.50 與 0.35 的門檻調整。
- Quiz 2、3 都是 seed 42、80%／20% 的分層切分；StandardScaler 只 fit 訓練資料。
- Quiz 2 的兩個模型各實際訓練完整 60 epochs，再以最低 validation BCE 選 checkpoint；分類指標使用最佳 checkpoint，loss 曲線則保留全部 60 epochs。
- 改良模型同時加入 Dropout 與 L2，報告比較整體策略，不把效果單獨歸因於其中一項。
- 類別欄位保留原始數值編碼，未額外 one-hot，是這份作業基準的簡化假設。
- 驗證資料參與 checkpoint 選擇，因此結果是 validation 成績，並非獨立 test 結果。
- AI 協作：ChatGPT／Codex 協助需求整理、程式、說明與執行驗證；所有圖表及數值均由實際執行產生。說明使用客觀觀察，未代寫個人學習經歷。

## 7. 作業要求核對

| 題目 | 講義要求 | 對應內容 |
|---|---|---|
| Quiz 1 基礎 | 二元分類、X 至少 5 欄 | 8 特徵、標籤 0／1 的 3,000 筆模擬資料 |
| Quiz 1 基礎 | Scikit-Learn、train／validation、scaling | Logistic Regression、80%／20% 切分、StandardScaler Pipeline |
| Quiz 1 基礎 | 機率、自訂及微調門檻、混淆矩陣、Accuracy／Precision／Recall／F1 | 0.50、0.35 與訓練 CV 門檻的完整表格及圖 |
| Quiz 1 進階 | 第二種模型、合適門檻、相同驗證資料、錯誤比較 | Random Forest、CV 門檻、同一驗證集、FP／FN 與指標說明 |
| Quiz 2 基礎 | 指定 Kaggle 信用卡資料、scaling、TensorFlow 或 PyTorch MLP | 原始 CSV、train-only StandardScaler、PyTorch MLP |
| Quiz 2 基礎 | 單筆 validation sample 預測 | 顯示原始特徵、真實標籤、模型機率與預測類別 |
| Quiz 2 基礎 | 至少 50 epochs、train／validation loss 圖、超參數解釋 | 原始 MLP 60 epochs、完整曲線、超參數表 |
| Quiz 2 進階 | 改善策略、重新訓練、loss 與四指標比較、觀察 | Dropout 0.25＋L2 0.001、改良 MLP 60 epochs、圖表、過擬合差距及波動分析 |
| Quiz 3 基礎 | 沿用 Quiz 2 資料及 MLP、機率、ROC 軸與 AUC、解釋 | 使用保存的切分與 Improved MLP 機率、MLP ROC、圖例 AUC、原理解釋 |
| Quiz 3 進階 | 課程算法、同一 train／validation、第二模型機率、同圖 ROC 及兩個 AUC、比較 | 講義第 31 頁 Random Forest、索引一致性檢查、同圖雙曲線、AUC 與局部 TPR 比較 |
