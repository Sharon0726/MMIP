# MMIP Week 4：時間序列模型與 Transformer

**蘇芯儀 / Sharon0726** · 課程日期：2026-09-30

四題作業包含基礎與進階：資料集準備、RNN / LSTM 情感分類、ViT / ResNet 微調，以及影像描述訓練、Gemini 評審、BLEU 與速度比較。

## 閱讀順序

1. [完整實驗報告](REPORT.md)
2. [Quiz 1：資料集準備](Quiz1_Datasets.ipynb)
3. [Quiz 2：情感分類](Quiz2_Sentiment.ipynb)
4. [Quiz 3：ViT 與 ResNet](Quiz3_Vision_Transformer.ipynb)
5. [Quiz 4：Image Captioning](Quiz4_Image_Captioning.ipynb)

四份閱讀版 Notebook 讀取保存的實際結果，不會自動重訓。訓練使用下方 Colab Runner。

## GPU 執行

[在 Colab 開啟完整訓練 Notebook](https://colab.research.google.com/github/Sharon0726/MMIP/blob/main/Week4/MMIP_Week4_Colab.ipynb)，選「執行階段 → 變更執行階段類型 → T4 GPU」，再依序執行。

Notebook 內嵌原始碼，無須 GitHub 憑證。完成後下載 `MMIP_Week4_results.zip`，含結果、圖表與權重。第一次執行需下載資料集及公開預訓練權重。Gemini 評審使用網頁版；重做方法與原始回覆保留於報告及 `results/gemini_review/`。

本機可檢查結果，不需要 GPU。使用 Python 3.12，先安裝適合平台的 PyTorch / torchvision，再執行：

```bash
pip install -r requirements.txt
python verify_results.py
python build_report.py
python build_notebooks.py
```

CPU 也能訓練，但完整實驗較慢：`python run_all.py`。訓練會重寫同名結果；如需保留既有實驗，請先複製整個資料夾。

## 方法與資料

| 題目 | 資料 | 方法 | 評估 |
|---|---|---|---|
| Quiz 1 | Rotten Tomatoes、CIFAR-10、Flickr8k | 固定切分、訓練集字典、來源與樣本數 | 切分清單與去重檢查 |
| Quiz 2 | 英文影評二元情感分類 | Embedding + RNN / LSTM | Accuracy、Macro-F1、Precision、Recall、ROC-AUC |
| Quiz 3 | CIFAR-10 十類 | 微調 ViT-Tiny 最後一區塊、ResNet18 layer4 與分類層 | 全部官方 test 10,000 張，OvR Macro-AUC |
| Quiz 4 | Flickr8k 影像與五個英文描述 | 凍結 ResNet18 + 訓練 GRU / LSTM decoder | 全部保留 test 影像的 corpus BLEU-1/4、100 張單張計時、Gemini 抽樣評審 |

CIFAR-10 訓練使用每類 1,000 張、驗證每類 200 張；其餘官方訓練影像不使用。Flickr8k 使用 seed=42 的自訂影像層級切分，不宣稱採用官方或 Karpathy 切分。模型以驗證集選 checkpoint；測試集不參與調參。

## 可查核檔案

- `results/*_split.csv`：每筆資料所屬切分及來源 ID。
- `results/*_metrics.csv`：量化結果；`results/predictions/`：逐筆測試預測。
- `results/history/`：每個 epoch 的訓練／驗證結果。
- `results/environment.json`、`config.json`、`results/source_sha256.json`、`results/training.log`：執行環境與來源追蹤。
- `results/verification.json`：從保存的預測重新計算指標的檢查結果。
- `checkpoints/`：本機與 Colab 匯出檔保留權重；大型權重、完整原始資料與講義不提交 Git。

## 資料來源與限制

- [Cornell Rotten Tomatoes dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)：Pang & Lee (ACL 2005) 英文影評；資料卡未明確標示授權，本作業保留下載程式、測試句及其推論紀錄，不打包完整原始資料，請遵守原始用途與權利。
- [CIFAR-10 官方資料集](https://www.cs.toronto.edu/~kriz/cifar.html)：Krizhevsky，32×32 RGB、十類；下載使用 [多倫多大學 HF 鏡像](https://huggingface.co/datasets/uoft-cs/cifar10)。
- [Flickr8k 下載封裝](https://huggingface.co/datasets/tsystems/flickr8k)：每張影像多個描述。封裝標示 MIT 不代表原始照片權利一律相同；僅保留作業評估必要的小量例圖。
- [ViT-Tiny 預訓練模型](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)、[torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。

ViT 的預訓練資料規模大於 ResNet，因此結果代表此設定下的實用比較，不能單獨證明架構優劣。單一 seed、有限訓練集與小量 Gemini 評審，也不能代表所有資料分布。
