# MMIP Week 3：CNN 影像分類作業

**蘇芯儀 / Sharon0726** · Multimodal Image Processing · 課程日期：2026-09-23

以 CIFAR-10 完成自訂 Plain CNN、ResNet-18 Transfer Learning、超參數實驗、資料擴增、ROC／Macro-AUC、兩個 Kernel 視覺化與 Grad-CAM。所有數值與圖片由實際訓練後的 checkpoints 產生。

## 閱讀順序

1. [完整報告 / Report](REPORT.md)：實驗設計、結果與逐項分析。
2. [Quiz 1：資料集](Quiz1_Dataset.ipynb)：來源、問題定義、固定切分。
3. [Quiz 2：CNN 訓練與比較](Quiz2_CNN_Training.ipynb)：模型、超參數、測試預測、ROC 與參數量。
4. [Quiz 3：泛化與解釋](Quiz3_Generalization_XAI.ipynb)：擴增前後比較、Kernel 與 Grad-CAM。

三份 Notebook 附已執行的輸出，可直接在 GitHub 閱讀；重新執行 Notebook 會讀取提交的實驗結果，不會自動重新訓練。

## 作業要求對照

| 項目 | 配分 | 交付證據 |
|---|---:|---|
| 資料來源、切分、分類任務 | 15 | Quiz 1、`results/split_manifest.csv` |
| Plain CNN 訓練與 Top-1／Top-5 | 9 | Quiz 2、`models.py`、訓練曲線、checkpoints |
| Plain CNN Testing Dataset 預測 | 3 | `results/predictions/plain_*.csv` |
| 經典 CNN Transfer Learning 與 Top-1／Top-5 | 9 | Quiz 2、ResNet-18、訓練曲線、head checkpoints |
| 經典 CNN Testing Dataset 預測 | 3 | `results/predictions/resnet_*.csv` |
| 各類 ROC、Macro-AUC、Accuracy／參數量比較 | 6 | `results/figures/roc_*.png`、`results/metrics.csv` |
| Plain CNN 超參數實驗 | 10 | 兩個 learning rates、固定其餘設定 |
| 經典 CNN 超參數實驗 | 10 | 兩個 learning rates、固定其餘設定 |
| Plain CNN 資料擴增比較 | 8 | Quiz 3、baseline／augmentation 對照 |
| 經典 CNN 資料擴增比較 | 7 | Quiz 3、baseline／augmentation 對照 |
| 兩個訓練後 Kernel 視覺化與分析 | 10 | `two_kernels.png`、`kernel_probes.png` |
| XAI 預測解釋與關注區域分析 | 10 | `gradcam_examples.png`、`results/xai_cases.csv` |
| **總計** | **100** | 全部基礎與進階項目 |

## 執行方式

主要訓練使用 Google Colab GPU。開啟 [Colab 訓練 Notebook](https://colab.research.google.com/github/Sharon0726/MMIP/blob/main/Week3/MMIP_Week3_Colab.ipynb)，或上傳本目錄的 `MMIP_Week3_Colab.ipynb`，選 GPU runtime 後依序執行。此 Notebook 已內嵌完整程式，不需先取得其他 GitHub 檔案。

實驗使用完整官方訓練資料，分成 45,000 train／5,000 validation，另使用官方 10,000 test。實際 GPU 與版本見 `results/environment.json`。

本機重現可使用 Python 3.12；CPU 也可執行，但會較慢。於本目錄執行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe analyze_results.py
.\.venv\Scripts\python.exe verify_results.py
```

Linux／macOS 將 `.\.venv\Scripts\python.exe` 換成 `.venv/bin/python`；依平台安裝相容的 PyTorch。不同平台可能有浮點數差異。

`run_experiments.py` 會下載 CIFAR-10 與官方 ResNet-18 ImageNet 權重。資料切分與種子在 `experiment_config.json`。完整訓練包含 3 次 Plain CNN × 15 epochs 及 3 次 ResNet head × 25 epochs。資料、ImageNet 權重及快取不放入 Git；需要網路進行首次下載。

已完成且設定相同的 run 會被重用。若要強制重新訓練，先將 `checkpoints/` 與 `results/runs/` 搬到備份位置，再執行。調整實驗設定時，請使用另一份作業目錄，以保留本次繳交的原始結果。

重新產生 Notebook（讀取現有報告與結果，並實際執行所有 cells）：

```powershell
.\.venv\Scripts\python.exe build_notebooks.py
```

單張影像推論：

```powershell
.\.venv\Scripts\python.exe predict.py --model plain --image path/to/image.png
.\.venv\Scripts\python.exe predict.py --model resnet --image path/to/image.png
```

ResNet checkpoint 僅儲存已訓練的分類 head；`predict.py` 會先載入官方 `IMAGENET1K_V1` 凍結 Backbone，再接回該 head，組成完整模型。

## 檔案與重現證據

| 路徑 | 內容 |
|---|---|
| `models.py` | Plain CNN 與 frozen ResNet-18 定義 |
| `run_experiments.py` | 資料、訓練、選模與測試的完整流程 |
| `prepare_data.py` | 官方資料的分段續傳，並核對完整 archive checksum |
| `analyze_results.py` | ROC、圖表、Kernel probes、Grad-CAM |
| `verify_results.py` | 指標獨立重算、資料切分、checkpoint 與推論一致性檢查 |
| `checkpoints/*.pt` | 全部 6 次實驗的最佳驗證 checkpoint |
| `results/history/*.csv` | 每個 epoch 的 loss 與 accuracy |
| `results/predictions/*` | 每次實驗完整 10,000 張測試預測及 10 類機率 |
| `results/selection.json` | 測試推論前鎖定的驗證集選模結果 |
| `results/environment.json` | 套件版本、時間、設定與預訓練權重 SHA-256 |
| `results/source_sha256.json` | Colab 實際執行的程式與設定檔雜湊 |
| `results/verification.json` | 實際驗證結果 |

`requirements-local-lock.txt` 記錄本機整理與 Notebook 驗證環境；GPU 訓練實際使用的版本以 `results/environment.json` 為準。

PDF 教材及原始 `md檔/` 資料夾只保留在本機，已於 `.gitignore` 排除。本 README 與 REPORT 是本次作業新撰寫的說明。

## Sources

- CIFAR-10：[資料集官方頁](https://www.cs.toronto.edu/~kriz/cifar.html)；Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*。
- ResNet：[He et al. (2015)](https://arxiv.org/abs/1512.03385)；[TorchVision ResNet-18](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html)。
- Grad-CAM：[Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391)。
- 課程：鄭中嘉（Eric），MMIP Week 03，2026-09-23，第 79–81 頁作業要求（本機教材，不隨作業上傳）。
