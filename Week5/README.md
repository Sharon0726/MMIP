# MMIP Week 5：多模態深度學習神經網路

**蘇芯儀 / Sharon0726** · 課程日期：2026-10-07

依講義第 54～56 頁的 **Quiz 1～3** 完成基礎及進階要求：影像檢索、多模態分類、CLIP Image Captioning。所提供講義及三張題目圖片沒有 Quiz 4。

## 閱讀與重現

- [完整實驗報告](REPORT.md)
- [Quiz 1：CNN 與 ViT 影像檢索](Quiz1_Image_Retrieval.ipynb)
- [Quiz 2：圖文融合分類](Quiz2_Multimodal_Classification.ipynb)
- [Quiz 3：CLIP 影像描述與 Week4 比較](Quiz3_CLIP_Captioning.ipynb)
- [在 Colab 開啟 GPU 訓練入口](https://colab.research.google.com/github/Sharon0726/MMIP/blob/main/Week5/MMIP_Week5_Colab.ipynb)

三份題目 Notebook 顯示已保存的實際結果，可在沒有 GPU 的環境閱讀。Colab Runner 是供重跑的乾淨入口，選 T4 GPU 後依序執行；程式內嵌，Week4 基準從固定 commit 下載並核對 SHA-256，不需要 GitHub 金鑰。下載資料與預訓練模型需網路。

本機檢查（Python 3.12；先安裝相符的 PyTorch / torchvision CPU 版本）：

```bash
pip install -r requirements.txt
python -m unittest test_invariants -v
python verify_results.py
python build_report.py
python build_notebooks.py
```

`python run_all.py` 會重新訓練並改寫同名結果；請先備份既有實驗。CPU 可執行，但完整訓練建議使用 Colab。

## 題目對照

| 題目 | 基礎 | 進階 |
|---|---|---|
| Quiz 1（30 分） | CIFAR-10 檢索資料、訓練 embedding、展示 top-5 | 分別微調 ResNet18 / ViT-Tiny，以 Hit@K、P@5、mAP@10 比較 |
| Quiz 2（35 分） | Flickr8k 圖片與文字、雙 Backbone 特徵、融合分類訓練與推論 | 更換影像 Backbone，重訓 ResNet+MiniLM 與 ViT+MiniLM；另做單模態對照 |
| Quiz 3（30 分） | 沿用 Week4 的 Flickr8k 切分與字典，CLIP encoder + 訓練文字 decoder | 四個 caption 系統用相同 test、五個參考句、BLEU 及 BERTScore 比較 |

Quiz 2 的類別定義是「原始圖文配對 / 同集合內打亂配對」，不是從 caption 猜出的物件類別。每張圖與每句話在正負樣本各出現一次，避免只靠某一模態的分布辨識標籤。不同圖片的描述仍可能語意相容，因此負例有雜訊；本作業不把這個標籤解讀為人類判定的語意正誤。

## 可查核資料

- `results/*_metrics.csv`、`caption_comparison.csv`：實際測試表現。
- `results/predictions/`：top-10 檢索 ID／相似度、分類機率、生成描述、逐張 BERTScore。
- `results/history/` 與 `results/training.log`：epoch 紀錄及 Colab 原始輸出。
- `results/retrieval_split.csv`、`flickr_split.csv`、`multimodal_pairs.csv`：分割與配對清單。
- `config.json`、`results/environment.json`、`results/source_sha256.json`：超參數、實際套件與原始碼雜湊。
- `baseline_week4/`：來自 [Week4 固定版本](https://github.com/Sharon0726/MMIP/tree/fe4592d6dd33388d3486c227a63eefc157843e88/Week4) 的測試預測、切分、字典與 BLEU；不重新生成或挑選舊模型結果。
- `results/verification.json`：從原始預測重算指標的查核。BERTScore 的本機檢查重算逐張平均，沒有宣稱重新執行神經網路。

完整資料集、特徵快取、講義與大型 checkpoint 不提交 Git。權重保留於本機 `checkpoints/` 及 Colab 結果匯出檔。

## 使用已訓練模型推論

將 Colab 匯出檔中的 `checkpoints/` 放回 Week5 後，可在 CPU 或 GPU 上使用 [predict.py](predict.py)：

```bash
python predict.py retrieval your_image.jpg --backbone vit
python predict.py multimodal your_image.jpg --text "a child is playing outside" --backbone resnet
python predict.py caption your_image.jpg --decoder gru
```

檢索輸出 CIFAR-10 gallery 的官方 train ID、類別與 cosine 分數；圖文分類輸出原始配對機率；caption 輸出自行訓練的 decoder 產生的英文描述。權重說明見 [checkpoints/README.md](checkpoints/README.md)。

## 來源與比較限制

- [CIFAR-10 官方說明](https://www.cs.toronto.edu/~kriz/cifar.html)、[UofT Hugging Face 鏡像](https://huggingface.co/datasets/uoft-cs/cifar10)。本次是同類別語意檢索，並非同一物體實例的檢索。
- [Flickr8k 封裝](https://huggingface.co/datasets/tsystems/flickr8k)。影像層級自訂 6000/1000/1091 切分完全沿用 Week4，並非官方標準切分；封裝授權不代表所有原始照片的權利相同。
- [Supervised Contrastive Learning 官方實作](https://github.com/HobbitLong/SupContrast)：本次使用排除自身的一視角 supervised contrastive loss，加上分類輔助損失。
- [ViT-Tiny 預訓練模型](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)、[torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。ViT 使用較多預訓練資料，結果不能單獨歸因於架構。
- [MiniLM 文字 Backbone 與 mean pooling](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。語言模型固定，訓練影像／文字投影及融合分類器。
- [CLIP ViT-B/32 模型卡](https://huggingface.co/openai/clip-vit-base-patch32)、[CLIP 文件](https://huggingface.co/docs/transformers/model_doc/clip)。只用 CLIP 影像塔及其視覺投影，文字由本次訓練的 GRU/LSTM 生成。
- [BERTScore 官方實作](https://github.com/Tiiiger/bert_score)：指定 BERT-base 第 9 層，所有方法一致使用多參考、無 IDF、無 baseline rescaling。分數不宜直接和其他 backbone／設定的 BERTScore 比較。

所有架構比較皆為單一 seed。預訓練資料、影像前處理與可訓練參數量不完全相同；CLIP 大規模網路預訓練與 Flickr8k 的潛在重疊無法在本作業中排除。報告同時保留成功及錯誤案例。
