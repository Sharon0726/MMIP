"""Build and execute three readable notebooks from the measured GPU artifacts."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
MD=nbf.v4.new_markdown_cell
CODE=nbf.v4.new_code_cell
SETUP='''from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
R=Path('results')
assert R.exists(), 'Run from Week5 with results/ present.'
def read(name): return json.loads((R/name).read_text(encoding='utf-8'))
def show(name): display(Image(filename=str(R/'figures'/name)))
def source(name): display(Markdown('```python\\n'+Path(name).read_text(encoding='utf-8')+'\\n```'))
'''


def make(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}})
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.validate(nb); nbf.write(nb,ROOT/name)
    print('Executed',name)


def main():
    intro='\n\n**蘇芯儀 / Sharon0726**\n\n本 Notebook 已執行並顯示 Colab T4 訓練保存的真實結果。訓練程式與推論程式完整附在同目錄；需要重訓請使用 `MMIP_Week5_Colab.ipynb`，本 Notebook 重跑只讀取結果，不重新訓練。'
    make('Quiz1_Image_Retrieval.ipynb',[
        MD('# Quiz 1：影像檢索（30 分）'+intro),CODE(SETUP),
        MD('## 1. 準備資料集（5 分）\n\nCIFAR-10：相同類別視為相關影像。官方 train 取每類 1000 訓練、200 驗證、1000 gallery，剩餘不使用；官方 test 全部作為 query。影像來源 ID 加上官方 split 才是唯一識別碼。影像為 224×224，只有訓練隨機水平翻轉。'),
        CODE("display(read('retrieval_data.json'))\ns=pd.read_csv(R/'retrieval_split.csv')\ndisplay(s.groupby(['split','label']).size().unstack())\nassert not s.duplicated(['source','source_id']).any()\nprint('PASS: no image ID crosses partitions.')"),
        MD('## 2. CNN 與 ViT 模型訓練（基礎 10 分、進階 5 分）\n\nResNet18 / ViT-Tiny → 128 維 L2 normalized embedding。部分微調 layer4 / 最後 Transformer block，使用 supervised contrastive loss + 0.5 × 分類 cross-entropy。6 epochs，AdamW lr=0.0003，batch=64；以 validation leave-one-out Hit@1 選 checkpoint。下面是實際模型及損失程式。'),
        CODE("src=Path('models.py').read_text(encoding='utf-8')\ndisplay(Markdown('```python\\n'+src[:src.index('class FusionModel')]+'\\n```'))\ndisplay(pd.read_csv(R/'retrieval_metrics.csv')[['model','best_epoch','trainable_parameters','backbone_max_abs_change','train_seconds']])\nshow('retrieval_learning.png')"),
        MD('完整訓練／推論迴圈在 [quiz1_retrieval.py](quiz1_retrieval.py)。Colab 原始輸出保存在 `results/training.log`，可核對各 epoch。'),
        MD('## 3. 展示檢索推論（5 分）\n\n每列為固定的前六張 test query 之一，以及 cosine similarity top-5 gallery 影像。綠字代表同類，紅字異類；沒有挑選成功例。'),
        CODE("show('retrieval_resnet.png')\nshow('retrieval_vit.png')\np=np.load(R/'predictions/retrieval_vit.npz')\ndisplay(pd.DataFrame({'gallery_train_id':p['gallery_ids'][0], 'label':p['neighbor_labels'][0], 'cosine':p['similarities'][0]}))"),
        MD('## 4. 比較檢索表現（5 分）\n\nHit@K 是 top-K 至少一張同類的查詢比例；P@5 是五張結果中同類比例。AP@10 的分母為 min(相關總數,10)=10，最後對 query 平均。此處 Hit@K 不能解讀為找回全部相關照片的 recall。'),
        CODE("d=pd.read_csv(R/'retrieval_metrics.csv')\ndisplay(d)\nshow('retrieval_comparison.png')\nbest=d.loc[d.map_at_10.idxmax()]\nprint(f'本次 mAP@10 較高：{best.model}，{best.map_at_10:.4f}')"),
        MD('ViT 的 ImageNet-21k+1k 與 ResNet 的 ImageNet-1k 預訓練不同，可訓練參數量也不同；單次 seed 比較不能歸因於架構本身。查詢時間包含單圖前處理與 encoder/top-10、排除磁碟讀圖及預先建立的 gallery；暖機 3 張後量測固定 100 張，前後同步 CUDA。完整限制與數據解讀見 [報告](REPORT.md)。')])
    make('Quiz2_Multimodal_Classification.ipynb',[
        MD('# Quiz 2：多模態深度學習神經網路（35 分）'+intro),CODE(SETUP),
        MD('## 1. 影像與文字訓練資料（5 分）\n\nFlickr8k 沿用 Week4 的 6000/1000/1091 影像切分。每張圖取第一句 caption；原始圖文配對標記 1，同集合內一對一打亂的文字標記 0。排除自身 ID 及相同文字。每張圖與每句話正負各一次，避免單一模態的分布洩漏標籤。不同照片也可能具有相容描述，因此這是原始配對分類，負例不是人類認定的語意錯誤。'),
        CODE("pairs=pd.read_csv(R/'multimodal_pairs.csv')\ndisplay(pairs.groupby(['split','label']).size().unstack())\nfor split,part in pairs.groupby('split'):\n    assert (part.groupby(['image_id','label']).size()==1).all()\n    assert (part.groupby(['text_id','label']).size()==1).all()\nprint('PASS: each input appears once under each label in its split.')\ndisplay(pairs.head(6))"),
        MD('## 2. 雙 Backbone 特徵融合（10 分）\n\n影像：ResNet18 512 維或 ViT-Tiny 192 維。文字：MiniLM 384 維，attention-mask mean pooling + L2 normalize。凍結兩個 backbone，分別投影為 256 維的 u、v，再融合 `[u,v,|u-v|,u*v]`，MLP 輸出二類 logits。'),
        CODE("src=Path('models.py').read_text(encoding='utf-8')\ndisplay(Markdown('```python\\n'+src[src.index('class FusionModel'):]+'\\n```'))\nfor kind in ['resnet','vit','minilm']: display(read('features_'+kind+'.json'))"),
        MD('## 3. 訓練與測試推論（5 分）\n\n25 epochs、AdamW lr=0.001、batch=128、cross-entropy，以 validation Macro-F1 選 checkpoint。完整程式在 [quiz2_multimodal.py](quiz2_multimodal.py)。以下固定前三張測試圖各展示正／負配對，不挑例。'),
        CODE("show('multimodal_learning.png')\nshow('multimodal_resnet.png')\nshow('multimodal_vit.png')\ndisplay(pd.read_csv(R/'predictions/multimodal_vit_fusion.csv').head(6))"),
        MD('## 4. 更換 Backbone、重訓與比較（15 分）\n\n實際比較 ResNet18+MiniLM 與 ViT-Tiny+MiniLM，同一配對、超參數、切分與指標；使用原始 ImageNet 特徵，不沿用 Quiz 1 的微調權重。另訓練 image-only / text-only 各 10 epochs 檢查平衡設計，它們不是相同預算的架構競賽。'),
        CODE("d=pd.read_csv(R/'multimodal_metrics.csv')\ndisplay(d)\nshow('multimodal_comparison.png')\nshow('multimodal_roc.png')\nshow('multimodal_confusion_resnet_fusion.png')\nshow('multimodal_confusion_vit_fusion.png')\nbest=d[d['mode']=='fusion'].sort_values('macro_f1',ascending=False).iloc[0]\nprint(f'融合組合 Macro-F1 較高：{best.model}，{best.macro_f1:.4f}')"),
        MD('ROC-AUC 使用連續正例機率，不能用 hard label 代替。單模態輸入的正負分布完全相同，test Accuracy 應為 0.5；Macro-F1 仍受預測正負比例影響。模型機率未校準，也不是新圖片描述是否真實的判定工具。見 [完整報告](REPORT.md)。')])
    make('Quiz3_CLIP_Captioning.ipynb',[
        MD('# Quiz 3：CLIP Image Captioning（30 分）'+intro),CODE(SETUP),
        MD('## 1. 沿用 Week4 資料（5 分）\n\n完全相同 Flickr8k 切分、train-only 字典、五句參考與 test 順序。Week4 測試預測由固定 Git commit 保存原樣，未重挑樣本。'),
        CODE("display(read('caption_data.json'))\ns=pd.read_csv(R/'flickr_split.csv')\npd.testing.assert_frame_equal(s,pd.read_csv('baseline_week4/caption_split.csv'))\ndisplay(s.groupby('split').size())\nprint('PASS: exactly the Week4 image-level split.')"),
        MD('## 2. CLIP encoder + 文字 decoder（10 分）\n\nCLIP ViT-B/32 影像塔及 visual projection 輸出 512 維；固定 encoder，訓練影像投影與 256 維 embedding / hidden GRU 或 LSTM。decoder 架構與 Week4 相同，CLIP text encoder 不參與文字生成。'),
        CODE("source('caption_decoder.py')\ndisplay(read('features_clip.json'))"),
        MD('## 3. 模型訓練與描述生成（5 分）\n\n每張 train 圖展開五句 caption；teacher forcing、忽略 PAD 的 token cross-entropy，20 epochs、Adam lr=0.001、batch=128。依 validation token loss 選模型；test 從 BOS 自回歸 greedy decode 到 EOS，不讀取參考句。完整迴圈在 [quiz3_clip_caption.py](quiz3_clip_caption.py)。'),
        CODE("show('caption_learning.png')\nshow('caption_examples.png')\ndisplay(pd.DataFrame(read('caption_examples.json'))[['filename','clip_gru','clip_lstm']])"),
        MD('## 4. 與 Week4 比較 BLEU、BERTScore（10 分）\n\n四個模型同 1091 張 test、五句參考。NLTK corpus BLEU-1/4 無 smoothing、沿用 regex tokenizer。BERTScore 為固定 revision 的 BERT-base-uncased 第 9 層、無 IDF、無 baseline rescaling；各圖在五參考中分別取最大 P/R/F，再對圖片平均。所有模型設定完全相同。'),
        CODE("display(read('bertscore_settings.json'))\nd=pd.read_csv(R/'caption_comparison.csv').sort_values(['week','model'])\ndisplay(d)\nshow('caption_comparison.png')\ndisplay(pd.read_csv(R/'caption_paired_comparison.csv'))"),
        MD('Paired bootstrap 2000 次只描述測試影像抽樣變異，不能當作多 seed 訓練變異。BLEU 看詞序重疊；BERTScore 看 token 語意相似，兩者都不直接驗證圖片內容。CLIP 預訓練規模與影像前處理也不同，無法排除與 Flickr8k 重疊。請搭配保留的錯誤描述觀察。'),
        CODE("display(read('verification.json'))\ndisplay(read('environment.json'))\ndisplay(read('completed.json'))"),
        MD('完整實驗解讀、比較限制與來源見 [REPORT.md](REPORT.md)。輸出的逐筆預測與 BERTScore CSV 可獨立查核。')])


if __name__=='__main__': main()
