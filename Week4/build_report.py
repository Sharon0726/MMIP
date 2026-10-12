"""Generate Traditional-Chinese report using measured experiment outputs."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))


def table(frame,cols=None):
    return (frame[cols] if cols else frame).to_markdown(index=False,floatfmt='.4f')


def main():
    s=pd.read_csv(ROOT/'results/sentiment_metrics.csv')
    v=pd.read_csv(ROOT/'results/vision_metrics.csv')
    c=pd.read_csv(ROOT/'results/caption_metrics.csv')
    env=load('environment.json'); sd=load('sentiment_data.json'); vd=load('vision_data.json'); cd=load('caption_data.json')
    s_best=s.loc[s.macro_f1.idxmax()]; v_best=v.loc[v.auc.idxmax()]
    c_best=c.loc[c.bleu4.idxmax()]; fast=c.loc[c.mean_ms.idxmin()]
    cases=load('gemini_cases.json')
    review_path=ROOT/'results/gemini_review/review.md'
    review=review_path.read_text(encoding='utf-8') if review_path.exists() else 'Gemini 評審尚未完成；不可將此項標記為已完成。'
    out=f'''# MMIP Week 4：四題實驗報告

**蘇芯儀 / Sharon0726** · 課程：2026-09-30 · 實驗：2026-10-12

本報告數值由真實訓練後的測試預測重新計算。運算裝置：**{env['gpu']}**，PyTorch {env['packages']['torch']}。完整環境、原始紀錄及來源雜湊位於 `results/`。

## Quiz 1：資料集準備（10 分）

| 任務 | train | validation | test | 輸入與標籤 |
|---|---:|---:|---:|---|
| Sentiment Analysis | {sd['counts']['train']} | {sd['counts']['validation']} | {sd['counts']['test']} | 英文影評，0 負面／1 正面 |
| Image Classification | {vd['counts']['train']} | {vd['counts']['validation']} | {vd['counts']['test']} | CIFAR-10 RGB 圖片，10 類 |
| Image Captioning | {cd['counts']['train']} | {cd['counts']['validation']} | {cd['counts']['test']} | Flickr8k 圖片，各有多個英文參考描述 |

### 來源與切分

1. **Rotten Tomatoes**：採 Hugging Face Cornell 封裝的官方 train/validation/test；以小寫與空白正規化後的文字 SHA-256 去除完全相同的句子，優先保留 test，再保留 validation。這不等同語意去重。字典只由訓練集建立，最大長度 64，未知字為 `<unk>`，不足補 `<pad>`。
2. **CIFAR-10**：從官方 50,000 張 train，以 seed=42 每類抽 1,000 張訓練與互斥的 200 張驗證；其餘 {vd['train_unused']} 張未使用。完整官方 10,000 張 test 都參與評估。影像調為 224×224；訓練隨機左右翻轉，驗證與測試不使用隨機增強。
3. **Flickr8k**：採 `tsystems/flickr8k`，共 {sum(cd['counts'].values())} 張。先按影像 ID 切分，再展開每張的參考描述，因此同一張照片的不同句子不會跨集合。train 有 {cd['caption_counts']['train']} 組影像描述。這是自訂切分，並非官方標準切分。字典僅使用訓練描述，共 {cd['vocab_size']} 個詞元，序列最長 32（含 BOS/EOS）。

切分原始 ID、類別或檔名見 `results/sentiment_split.csv`、`vision_split.csv`、`caption_split.csv`。來源連結與授權說明見 [README](README.md#資料來源與限制)。

## Quiz 2：Sentiment Analysis（20 分）

### 模型與訓練

兩模型均為 128 維 Embedding → 單向、單層 128 hidden recurrent layer → dropout 0.3 → 二元分類。RNN 使用 tanh；LSTM 以輸入／遺忘／輸出閘控制記憶。取每句最後一個有效詞的 hidden state，避免 padding 污染分類位置。無預訓練詞向量。

相同 seed=42、train/val/test、batch=64、Adam lr=0.001、12 epochs、cross-entropy、gradient norm clipping=1。用 **validation Macro-F1** 最高的 checkpoint，然後一次評估 test。不同單元的參數量不完全相同，表中揭露。

{table(s,['model','best_epoch','parameters','accuracy','macro_precision','macro_recall','macro_f1','auc','train_seconds'])}

此實驗以 Macro-F1 比較，**{s_best['model'].upper()}** 較高（{s_best['macro_f1']:.4f}）。AUC 使用正面類別的連續預測機率，不使用硬標籤。結果來自單次訓練，未估計多種子的變異，不能保證每次 LSTM 都比 RNN 好。

![情感分類學習曲線](results/figures/sentiment_learning.png)
![情感分類 ROC](results/figures/sentiment_roc.png)

### 測試推論

下表固定取 test 前八筆；0=negative，1=positive。完整預測與每類機率位於 `results/predictions/sentiment_*.csv`。

{table(pd.read_csv(ROOT/'results/predictions/sentiment_lstm.csv').head(8),['id','text','true','predicted','p_1'])}

## Quiz 3：Vision Transformer（35 分）

### 實際微調範圍

- **ViT-Tiny / patch16 / 224**：ImageNet-21k 預訓練再經 ImageNet-1k 微調的權重；更新最後一個 Transformer block、最後 LayerNorm 與新的十類分類層。
- **ResNet18**：ImageNet-1k 權重；更新 layer4 及新的十類分類層。其餘層與 frozen BatchNorm 統計固定。

兩者的可訓練 backbone 都有更新（`backbone_max_abs_change > 0`），是部分 backbone 微調。共用 10,000 train、2,000 val、10,000 test；AdamW lr=0.0003、weight decay=0.01、batch=64、6 epochs，以 validation accuracy 選最佳 epoch。不同模型使用對應預訓練 normalization。GPU 使用自動混合精度。

{table(v,['model','best_epoch','trainable_parameters','accuracy','macro_f1','auc','train_seconds','backbone_max_abs_change'])}

Macro-AUC 定義：對十個類別分別做 one-vs-rest ROC-AUC，再取等權平均。**{v_best['model']}** 的 Macro-AUC 較高（{v_best['auc']:.4f}）。注意兩個模型的預訓練資料規模不同、可訓練參數量不同；此結果比較整體設定，不能把差異完全歸因於 Transformer 或 CNN 架構。

![ViT 與 ResNet 學習曲線](results/figures/vision_learning.png)
![影像分類 Macro ROC](results/figures/vision_roc.png)
![固定 test 前十二張推論](results/figures/vision_examples.png)

## Quiz 4：Image Captioning（35 分）

### 模型訓練與描述生成

ImageNet 預訓練 ResNet18 移除分類層，固定 encoder，抽取 512 維影像特徵。可訓練投影把影像特徵轉成 decoder 初始 hidden state；LSTM 另有 cell state 投影。比較單層 GRU 與 LSTM，皆為 256 維 embedding、256 hidden、dropout=0.3。encoder 固定，但 decoder 及影像投影確實以 Flickr8k 訓練。

訓練使用 teacher forcing：輸入 BOS 與前一個真實詞，預測下一詞。Cross-entropy 忽略 PAD；Adam lr=0.001、batch=128、20 epochs、gradient norm clipping=1，以 validation 每詞 loss 選 checkpoint。測試為 autoregressive greedy decoding，從 BOS 開始，不使用 test 參考句作輸入，遇 EOS 或長度上限停止。

### BLEU 與回覆時間比較

{table(c,['model','best_epoch','decoder_parameters','bleu1','bleu4','mean_ms','median_ms','p95_ms','train_seconds'])}

BLEU 使用 NLTK corpus BLEU，對 **{cd['counts']['test']} 張 test 影像**與各自全部參考句計算。轉小寫後用同一 regex tokenizer；BLEU-1 權重 (1,0,0,0)、BLEU-4 權重 (0.25,0.25,0.25,0.25)，含 brevity penalty，不做 smoothing，值域 0–1。BLEU 衡量詞序重疊，對合理同義句可能低估，因此另用 Gemini 做影像語意抽查。

計時使用同一 GPU、同一組固定前 100 張 test、batch=1。每個方法先暖機 3 張；每筆計時前後同步 CUDA。時間包含 resize/normalize、CPU→GPU、encoder 與完整 decoder，排除磁碟讀圖、模型下載、權重載入、網路及 Gemini 回覆。圖像已在記憶體，但影像特徵在每次計時中重新計算，並非直接使用快取特徵。計時原始樣本見 `caption_latency_*.csv`。

此設定下 **{c_best['model']}** 的 BLEU-4 較高（{c_best['bleu4']:.4f}）；**{fast['model']}** 的平均回覆時間較短（{fast['mean_ms']:.2f} ms）。模型生成句長可能影響延遲，`caption_metrics.csv` 另附平均詞數。這是一次 100 張順序量測，未重複多輪；小幅速度差異不應解讀為穩定優勢。

![Caption 學習曲線](results/figures/caption_learning.png)
![Caption BLEU 與耗時](results/figures/caption_comparison.png)
![固定 test 前六張的生成描述](results/figures/caption_examples.png)

### Gemini 判斷描述與影像是否相符

{review}

## 查核與重現

`verify_results.py` 獨立從逐筆預測重算 Accuracy、Macro-F1、Precision、Recall、ROC-AUC 與 corpus BLEU，確認兩模型使用相同 test IDs、機率和為 1、各集合互斥、生成文字非空，以及微調的 backbone 權重實際改變。檢查狀態：**{load('verification.json')['passed']}**。

完整參數見 `config.json`；重現入口：[Colab Runner](MMIP_Week4_Colab.ipynb)。CPU 不需要負擔本次 GPU 訓練。權重保留於本機與 Colab 匯出壓縮檔，大型資料與權重不放 GitHub。不同硬體或套件版本可能有浮點數差異。

## 作業要求對照

| 項目 | 分數 | 證據 |
|---|---:|---|
| 三種資料集準備 | 10 | Quiz 1，來源、切分清單、字典及例圖 |
| RNN 訓練與 test 推論 | 10 | Quiz 2，訓練紀錄、逐筆機率 |
| LSTM 訓練與 RNN 比較 | 10 | Quiz 2，Macro-F1、AUC、曲線 |
| ViT 微調與推論 | 20 | Quiz 3，實際 backbone 更新、test 圖與機率 |
| ResNet 微調與 Macro-AUC 比較 | 15 | Quiz 3，完整 10,000 張 test |
| Caption 模型訓練與生成 | 15 | Quiz 4，兩種 decoder、test 描述 |
| Gemini 影像與描述相符性判斷 | 5 | 保存的 Gemini 提示、原始回覆及截圖 |
| BLEU 量化 | 5 | 全部 caption test、多參考 corpus BLEU |
| 方法間回覆時間與表現比較 | 10 | GRU/LSTM 的 100 筆計時、BLEU 與分析 |

## 限制與改善方向

小型 decoder、固定 encoder 與 greedy decoding 容易生成常見句型，對人物數量、顏色、動作及空間關係可能錯誤。BLEU 和 Gemini 都不是人類標準答案；Gemini 抽查僅六張影像，不能外推成全測試集準確率。可再比較 attention decoder、beam search、BLIP 微調與多種子實驗，但這些不屬於本次已執行結果。
'''
    (ROOT/'REPORT.md').write_text(out,encoding='utf-8')
    print('Generated REPORT.md from measured results')


if __name__=='__main__': main()
