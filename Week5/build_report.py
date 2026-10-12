"""Generate the Traditional Chinese report strictly from measured result files."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
def table(df,cols): return df[cols].to_markdown(index=False,floatfmt='.4f')


def main():
    r=pd.read_csv(ROOT/'results/retrieval_metrics.csv'); m=pd.read_csv(ROOT/'results/multimodal_metrics.csv')
    c=pd.read_csv(ROOT/'results/caption_comparison.csv').sort_values(['week','model'])
    ci=pd.read_csv(ROOT/'results/caption_paired_comparison.csv'); env=read('environment.json')
    rb=r.loc[r.map_at_10.idxmax()]; mb=m[m['mode']=='fusion'].sort_values('macro_f1',ascending=False).iloc[0]
    comparisons=[]
    for kind in ['gru','lstm']:
        old=c[c.model==f'resnet_{kind}'].iloc[0]; new=c[c.model==f'clip_{kind}'].iloc[0]
        comparisons.append({'decoder':kind,'BLEU4_delta_CLIP_minus_Week4':new.bleu4-old.bleu4,'BERTScore_F1_delta':new.bertscore_f1-old.bertscore_f1})
    delta=pd.DataFrame(comparisons)
    out=f'''# MMIP Week 5 實驗報告

**蘇芯儀 / Sharon0726** · 課程：2026-10-07 · 實驗：2026-10-12

講義與附圖提供 Quiz 1～3，本報告完成三題基礎與進階要求（合計 95 分的列示項目）。實際運算使用 **{env['gpu']}**，PyTorch **{env['packages']['torch']}**。所有表格來自保存的測試預測與訓練紀錄，完整執行環境見 `results/environment.json`。

## Quiz 1：影像檢索（30 分）

### 資料與模型

使用 CIFAR-10 十類資料，定義「相同類別」為相關影像，例如 cat 查到另一張 cat 屬於正確。這是語意類別檢索，沒有主張辨認同一實例。從官方 50,000 張訓練影像，以 seed=42 每類選 1,000 train、200 validation、1,000 gallery，其餘 28,000 不使用；官方 10,000 張 test 全部作為 query。四個集合的來源 ID 互斥，gallery 沒有參與模型訓練。

| 集合 | 影像数 | 用途 |
|---|---:|---|
| train | 10,000 | 更新模型 |
| validation | 2,000 | 集合內 leave-one-out 檢索，選 checkpoint |
| gallery | 10,000 | 測試檢索資料庫，每類 1,000 張 |
| query | 10,000 | 完整官方 test，只用於最終評估 |

ResNet18 使用 ImageNet-1k 預訓練，更新 layer4；ViT-Tiny / patch16 / 224 使用 ImageNet-21k 再經 ImageNet-1k 微調的權重，更新最後 Transformer block 及 LayerNorm。凍結其餘參數與未訓練區塊的 BatchNorm 統計。兩者接一個 **128 維投影並 L2 normalize**，以 cosine similarity 檢索；另加十類分類頭輔助訓練。

損失為 supervised contrastive loss + 0.5 × cross-entropy。前者在同一 batch 中拉近同類影像，推遠異類影像，分子與分母皆排除影像自身，temperature=0.1。使用 AdamW lr=0.0003、weight decay=0.01、batch=64、6 epochs、梯度裁切 1、GPU 混合精度。圖片調為 224×224，只有 train 使用隨機水平翻轉。checkpoint 依 validation Hit@1 選取；儲存的 backbone tensor 確實有更新。

### 測試結果與解讀

{table(r,['model','best_epoch','trainable_parameters','hit_at_1','hit_at_5','hit_at_10','precision_at_5','map_at_10','mean_query_ms','train_seconds'])}

Hit@K 表示前 K 張中至少一張與 query 同類的查詢比例，**不是找回全部相關影像的 recall 比例**。P@5 是前五張中的相關比例。AP@10 = 前十名每個相關位置之 precision 的總和 ÷ min(相關總數,10)；此資料每類 gallery 有 1,000 張，分母為 10。mAP@10 再對全部 query 取平均。上述指標都以連續 cosine 排名計算。

本次 **{rb['model']}** 的 mAP@10 較高（{rb.map_at_10:.4f}）。比較的是這組預訓練與部分微調設定；ViT 預訓練資料較多，兩個模型可訓練參數量也不同，不能將差異完全歸因於 CNN 或 Transformer。

查詢耗時在同一 GPU 上量測，batch=1，先暖機 3 張，再固定測試第 4～103 張共 100 張；計時前後同步 CUDA。包含影像前處理、CPU→GPU、encoder、與預先建好的 gallery 做 cosine 及 top-10；排除磁碟讀圖、資料庫建置與模型載入。耗時只做一次量測，不推定小差異為穩定優勢。

![檢索比較](results/figures/retrieval_comparison.png)
![檢索訓練曲線](results/figures/retrieval_learning.png)

以下固定選前六張 query；每列左側為查詢，其餘為 top-5。綠字同類、紅字異類，沒有按成功率挑樣本。

![CNN 檢索推論](results/figures/retrieval_resnet.png)
![ViT 檢索推論](results/figures/retrieval_vit.png)

## Quiz 2：多模態分類（35 分）

### 自行建立圖文資料與標籤

使用 Flickr8k 的真實影像與人工文字描述，完全沿用 Week4 的影像層級切分：6000 train、1000 validation、1091 test。每張圖取第一個 caption，建立「原始配對」正例；在同一 split 內用一對一 permutation 產生「非原始配對」負例，拒絕同圖配對及相同正規化文字。train/validation/test 分別為 **12000/2000/2182 組**，每類各一半。

同一張圖與同一句文字在正负樣本各出現一次。因此單看圖片或單看文字的邊際分布完全平衡，需要利用圖文關係。各 split 先按影像分開，負例不會借用另一集合的圖片或描述。

**標籤的限制：** 不同照片的 caption 仍可能描述相似場景，因此負例可能在語意上合理。此任务辨認資料集原始圖文配對，並不是人類逐筆認定的描述正誤。對新圖片的配對機率也沒有做校準。

### Backbone、融合與訓練

比較兩個重新訓練的組合：**ResNet18 + MiniLM**、**ViT-Tiny + MiniLM**。使用原始 ImageNet backbone，不沿用 Quiz 1 微調後權重。影像特徵分別為 512／192 維；MiniLM 使用 attention-mask mean pooling 與 L2 normalization，得到 384 維文字特徵，最多 128 word pieces。

兩種 backbone 都凍結，實際訓練影像／文字 LayerNorm + 線性投影到 256 維，得到 u、v；融合 `[u, v, |u−v|, u×v]` 後，接 256 hidden ReLU、dropout=0.3 與二類輸出。AdamW lr=0.001、weight decay=0.01、batch=128、25 epochs，cross-entropy，以 validation Macro-F1 選 checkpoint，test 門檻為兩類 argmax。單看影像與單看文字的兩個對照各訓練 10 epochs，不列為相同訓練預算的 backbone 競賽。

{table(m,['model','best_epoch','accuracy','macro_precision','macro_recall','macro_f1','auc','trainable_parameters','train_seconds'])}

兩個融合組合中，**{mb['model']}** 的 Macro-F1 較高（{mb.macro_f1:.4f}）。Accuracy、Macro-F1 與連續正例機率的 ROC-AUC 都保留供比較；不要只看最佳單一指標。訓練秒數不包含凍結 backbone 的特徵抽取，抽取時間另記於 `results/features_*.json`。

![多模態比較與單模態對照](results/figures/multimodal_comparison.png)
![多模態訓練曲線](results/figures/multimodal_learning.png)
![多模態 ROC](results/figures/multimodal_roc.png)

以下固定前六筆測試配對（前三張圖片的正負例），1=原始配對、0=打亂配對。

![ResNet 與 MiniLM 推論](results/figures/multimodal_resnet.png)
![ViT 與 MiniLM 推論](results/figures/multimodal_vit.png)

## Quiz 3：CLIP Image Captioning（30 分）

### 公平比較設定

沿用 Week4 完全相同的 **6000/1000/1091** 影像 ID、訓練字典、五個參考描述與測試順序。只有 train 描述建立字典，模型不以 test 描述作生成輸入。保存的 Week4 GRU/LSTM 逐筆預測直接作為基準，沒有重新生成、挑選或修改舊結果。

本次採 **CLIP ViT-B/32 固定影像 encoder + 視覺投影（512 維）**，以可訓練影像投影初始化 GRU 或 LSTM 的 hidden state（LSTM 另有 cell state），搭配 256 embedding、256 hidden 的單層 decoder。未使用 CLIP text encoder 生成文字。decoder 架構與 Week4 相同，訓練影像投影及全部 decoder 參數。

同為 20 epochs、Adam lr=0.001、batch=128、dropout=0.3、梯度裁切 1。使用 teacher forcing、忽略 PAD 的 token cross-entropy，以 validation token loss 選 checkpoint。推論用 autoregressive greedy decoding，BOS 開始、EOS 停止，長度最多 32（含 BOS/EOS），不允許生成 PAD/UNK/BOS。

CLIP 使用自身官方影像處理；Week4 ResNet 使用 torchvision 官方 resize/crop/normalization。因此本次比較代表 encoder 與相應前處理的整體效果。CLIP 的大量圖文預訓練可能與測試照片有潛在重疊，本作業無法排除。

本次 decoder 的實際訓練與模型選擇如下。驗證損失後段若回升，仍載入最低驗證損失那一輪的權重。

{table(c[c.week==5],['model','best_epoch','decoder_parameters','train_seconds'])}

### 全部 1091 張測試影像：BLEU 與 BERTScore

{table(c,['week','model','bleu1','bleu4','bertscore_precision','bertscore_recall','bertscore_f1','test_images'])}

BLEU-1/4 由 NLTK corpus BLEU 計算：與 Week4 相同的小寫 regex tokenizer，五個參考句，權重分別 (1,0,0,0) 與 (0.25,0.25,0.25,0.25)，包含 brevity penalty，不 smoothing；值域 0–1。

BERTScore 固定使用 `google-bert/bert-base-uncased` 第 9 層，bert-score 0.3.13、無 IDF、無 baseline rescaling、slow tokenizer、batch=32。每張候選句與五個參考句比對，依官方多參考實作分別對 precision/recall/F1 取最大，再對影像取平均。BERTScore Precision、Recall、F1 不是分类正确率，也不宜與使用其他語言模型或 rescaling 的數值直接比較。模型 revision 與 hash 完整記錄於 `results/bertscore_settings.json`。

同一 decoder 替換 encoder 的變化（CLIP 減 Week4）：

{delta.to_markdown(index=False,floatfmt='.4f')}

本次兩種 decoder 換成 CLIP 後，BLEU-4 與 BERTScore F1 都提高。GRU 的 BLEU-4 增加 {delta.iloc[0]['BLEU4_delta_CLIP_minus_Week4']:.4f}、BERTScore F1 增加 {delta.iloc[0]['BERTScore_F1_delta']:.4f}；LSTM 分別增加 {delta.iloc[1]['BLEU4_delta_CLIP_minus_Week4']:.4f}、{delta.iloc[1]['BERTScore_F1_delta']:.4f}。這是本次固定測試集與設定下的結果。

{ci.to_markdown(index=False,floatfmt='.4f')}

上表信賴區間是以同一張圖的 BERTScore F1 差值做 2000 次 paired bootstrap；描述的是這批 test 影像的抽樣變異，不涵蓋訓練 seed、預訓練資料重疊或模型選擇的不確定性。BLEU 看詞序重疊，BERTScore 看 contextual token similarity，兩者都不是圖片內容正確性的直接保證。

![兩週 Caption 指標比較](results/figures/caption_comparison.png)
![CLIP Caption 訓練曲線](results/figures/caption_learning.png)

固定 test 前六張的四種輸出如下。成功與失敗例都保留；流暢但說錯動作、數量或物體的描述仍是錯誤。

![四個 Caption 系統推論](results/figures/caption_examples.png)

固定案例的視覺檢查顯示，平均分數提升仍伴隨明顯錯誤：

| 案例 | 觀察 |
|---|---|
| 1：棒球 | CLIP-GRU 產生「three men play baseball」，辨識出運動種類；Week4 把它說成足球或籃球。CLIP-LSTM 雖提到棒球，卻捏造紅色球衣與傳球動作。 |
| 2：女子與地面粉筆畫 | CLIP-GRU 看到藍色衣服，但把畫中人物說成另一位女子；CLIP-LSTM 說成紅衣女子抱嬰兒。均未正確描述粉筆畫。 |
| 3：戴耳機的男子 | CLIP 輸出反覆的「a beard」，未描述耳機；顯示 greedy recurrent decoder 的重複問題。 |
| 4：橄欖球 | GRU 仍混淆成 soccer；LSTM 的 football 描述較概括，未說出爭球動作。 |
| 5：建物前的人群 | CLIP 說出人群與建物，但省略人物與場景細節。 |
| 6：淋浴中的男孩 | CLIP 把場景說成 pool，甚至捏造抱嬰兒或嘴裡含小孩；語句與圖片內容不符。 |

因此 CLIP 改善了整體詞彙／語意分數，並不代表已能可靠描述細節。這六例只是固定展示樣本，不能當作整體正確率。

## 查核與重現

`verify_results.py` 已從保存的逐筆輸出重算檢索指標、分類 Accuracy/F1/AUC、corpus BLEU，並核對 BERTScore 逐張分數的聚合、同一測試集、影像層級分割與每種輸入的標籤平衡。神經網路 BERTScore 在 GPU 實際計算，本機輕量驗證只重算平均；不把這兩者混為一談。查核結果：**{read('verification.json')['passed']}**。

程式及切分見 [README](README.md)，GPU 重跑入口為 [Colab Notebook](MMIP_Week5_Colab.ipynb)。原始碼 SHA-256、套件版本、模型 revision、原始 training.log 及逐筆預測皆已保存。checkpoint 留在本機與 Colab 匯出檔，不提交大型權重到 Git。

## 限制與改善

這些是單一 seed 的課程實驗，沒有多次訓練的誤差範圍。CIFAR-10 同類別檢索較粗糙，無法代表商用細粒度搜尋。原始圖文配對的負例有語意雜訊，未做人類核實。固定 encoder、小型 recurrent decoder 與 greedy decoding 可能省略細節或產生常見句型；後續可比較人工驗證的 hard negatives、attention decoder、beam search、多種子及去除近重複影像的測試，但這些不是本次已完成實驗。
'''
    out=out.replace('影像数','影像數').replace('正负','正負').replace('此任务','此任務').replace('分类正确率','分類正確率')
    (ROOT/'REPORT.md').write_text(out,encoding='utf-8')
    print('Generated REPORT.md from actual measurements')


if __name__=='__main__': main()
