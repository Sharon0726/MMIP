"""Build and EXECUTE four readable quiz notebooks from measured result artifacts."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
MD=nbf.v4.new_markdown_cell
CODE=nbf.v4.new_code_cell
SETUP='''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
R=Path('results')
assert R.exists(), 'Run from Week4; results/ is required.'
def read(name): return json.loads((R/name).read_text(encoding='utf-8'))
def show(name): display(Image(filename=str(R/'figures'/name)))
'''


def make(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}})
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.validate(nb); nbf.write(nb,ROOT/name)
    print('Executed',name)


def main():
    make('Quiz1_Datasets.ipynb',[
        MD('# Quiz 1：資料集準備（10 分）\n\n**蘇芯儀 / Sharon0726**\n\n三種任務的來源、切分、前處理及實例。訓練與下載程式在 `quiz2_sentiment.py`、`quiz3_vision.py`、`quiz4_caption.py`，完整 GPU 流程為 `MMIP_Week4_Colab.ipynb`。'),CODE(SETUP),
        MD('## 1. 情感分類資料（3 分）\n\n[Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)：英文影評，negative=0、positive=1。保留官方切分，去除正規化後的相同句子，字典只取 train；截斷至 64 tokens 並 padding。'),
        CODE("display(read('sentiment_data.json'))\ndisplay(pd.read_csv(R/'sentiment_split.csv').groupby(['split','label']).size().unstack())\ndisplay(pd.read_csv(R/'predictions/sentiment_lstm.csv')[['text','true']].head(6))"),
        MD('## 2. 影像分類資料（4 分）\n\n[CIFAR-10 官方來源](https://www.cs.toronto.edu/~kriz/cifar.html)，使用 [UofT HF 鏡像](https://huggingface.co/datasets/uoft-cs/cifar10)。train/val 每類 1,000/200，其餘 train 不使用；test 為完整官方 10,000 張。模型輸入 resize 到 224×224，只有 train 隨機左右翻轉。'),
        CODE("display(read('vision_data.json'))\ndisplay(pd.read_csv(R/'vision_split.csv').groupby(['split','label']).size().unstack())\nshow('vision_examples.png')"),
        MD('## 3. 影像文字配對資料（3 分）\n\n[Flickr8k](https://huggingface.co/datasets/tsystems/flickr8k)：先以影像 ID 切 train/val/test，再展開每張的多個 caption；不讓同圖不同 caption 跨集合。自訂 seed=42 切分，非官方 Karpathy split。只由 train 建字典，長度 32 含 BOS/EOS。'),
        CODE("display(read('caption_data.json'))\ncase=read('gemini_cases.json')[0]\nshow('caption_sample_1.jpg')\ndisplay(case['references'])"),
        MD('## 資料分離檢查'),CODE("s=pd.read_csv(R/'sentiment_split.csv'); assert s.sha256.is_unique\nv=pd.read_csv(R/'vision_split.csv'); assert not set(v.query('split == \"train\"').source_id) & set(v.query('split == \"validation\"').source_id)\nc=pd.read_csv(R/'caption_split.csv'); assert c.filename.is_unique\nprint('PASS: normalized text and image IDs do not cross partitions.')")])
    make('Quiz2_Sentiment.ipynb',[
        MD('# Quiz 2：Sentiment Analysis（20 分）\n\n**蘇芯儀 / Sharon0726**\n\n建立與訓練 RNN（5）、test 推論（5）、LSTM（5）、兩模型評估比較（5）。'),CODE(SETUP),
        MD('## 模型\n\n128 維 embedding → 128 hidden RNN 或 LSTM → dropout → 二類分類。取最後有效詞，不取補零位置。使用同一訓練切分、12 epochs、Adam lr=0.001；以 validation Macro-F1 選 checkpoint。'),
        CODE("source=Path('models.py').read_text(encoding='utf-8')\ndisplay(Markdown('```python\\n'+source[source.index('class SentimentModel'):source.index('class CaptionModel')]+'```'))"),
        MD('## 實際訓練紀錄與評估'),CODE("display(pd.read_csv(R/'sentiment_metrics.csv'))\nshow('sentiment_learning.png')\nshow('sentiment_roc.png')"),
        MD('需要重做訓練時，可在 Colab Runner 執行全部實驗，或在已安裝相依套件與備份結果的 Week4 目錄執行 `python quiz2_sentiment.py`。訓練迴圈含 loss.backward、gradient clipping、optimizer.step 與 validation checkpoint selection；程式完整保留在同名檔案。'),
        MD('## 使用測試資料推論\n\n0=負面，1=正面；下表為固定 test 前十二筆，完整預測已存 CSV。'),
        CODE("for model in ['rnn','lstm']:\n    print(model.upper())\n    display(pd.read_csv(R/f'predictions/sentiment_{model}.csv').head(12))"),
        CODE("show('sentiment_rnn_confusion.png')\nshow('sentiment_lstm_confusion.png')"),
        MD('## 解讀\n\nAccuracy 表示分類正確率；Macro-F1 平等重視正負兩類；ROC-AUC 衡量連續正面機率的排序能力。僅有一個 seed，且兩種 recurrent unit 參數量不同，不宜將結果視為架構的普遍結論。完整數據分析見 REPORT.md。')])
    make('Quiz3_Vision_Transformer.ipynb',[
        MD('# Quiz 3：Vision Transformer（35 分）\n\n**蘇芯儀 / Sharon0726**\n\nViT 微調與推論（20）、ResNet 微調與推論（10）、Macro-AUC 比較（5）。'),CODE(SETUP),
        MD('## 微調設計\n\nViT-Tiny 更新最後 Transformer block、LayerNorm、head；ResNet18 更新 layer4 與 fc。其餘層凍結。相同 train/val/test、6 epochs、AdamW lr=0.0003，以 validation accuracy 選 checkpoint。ViT 預訓練使用 ImageNet-21k + 1k，ResNet 使用 ImageNet-1k，這是比較限制。'),
        CODE("source=Path('models.py').read_text(encoding='utf-8')\ndisplay(Markdown('```python\\n'+source[source.index('def make_vision'):]+'```'))"),
        CODE("m=pd.read_csv(R/'vision_metrics.csv'); display(m)\nassert (m.backbone_max_abs_change>0).all()\nprint('PASS: both pretrained backbones were updated.')\nshow('vision_learning.png')"),
        MD('重新微調入口：`python quiz3_vision.py`。需要 GPU 時使用同目錄 Colab Runner；預設不在這份結果閱讀版 Notebook 自動重訓。'),
        MD('## 完整 test 推論與 Macro-AUC\n\n十類各自視為 positive 做 one-vs-rest ROC，再對十個 AUC 等權平均。測試包含全部 10,000 張，兩模型完全相同的 ID。'),
        CODE("show('vision_roc.png')\nshow('vision_examples.png')\nfor model in ['vit','resnet']:\n    print(model)\n    display(pd.read_csv(R/f'predictions/vision_{model}.csv').head(5))"),
        CODE("show('vision_vit_confusion.png')\nshow('vision_resnet_confusion.png')")])
    make('Quiz4_Image_Captioning.ipynb',[
        MD('# Quiz 4：Image Captioning（35 分）\n\n**蘇芯儀 / Sharon0726**\n\n模型訓練與生成（15）、Gemini 判斷（5）、BLEU（5）、回覆時間與表現比較（10）。'),CODE(SETUP),
        MD('## 模型訓練\n\n固定 ImageNet ResNet18 encoder，訓練影像投影及 GRU / LSTM decoder，hidden/embedding=256。train 五個參考句皆參與 teacher forcing，test 為 greedy 自回歸生成。20 epochs，以 validation token loss 選 checkpoint。'),
        CODE("source=Path('models.py').read_text(encoding='utf-8')\ndisplay(Markdown('```python\\n'+source[source.index('class CaptionModel'):source.index('def make_vision')]+'```'))\nshow('caption_learning.png')"),
        MD('重新訓練與生成入口：`python quiz4_caption.py`；新圖片推論：`python predict.py caption image.jpg --model lstm`（需 checkpoints）。'),
        MD('## test 生成、BLEU 與計時\n\nBLEU-1/4 由 test 全部影像的所有參考句計算，為 corpus 分數（0–1），含 brevity penalty、不 smoothing。計時同一組 100 張、batch=1、暖機與 CUDA 同步，包含前處理 + encoder + decoder，排除磁碟讀圖及網路。'),
        CODE("display(pd.read_csv(R/'caption_metrics.csv'))\nshow('caption_comparison.png')\nshow('caption_examples.png')\ndisplay(pd.DataFrame(read('predictions/caption_lstm.json')).head(6))"),
        MD('## Gemini 實際評審\n\n固定前六張 test 影像，模型 A/B 盲評，附原始提示、回覆與截圖。小樣本僅作質性分析，不能視為整個測試集的準確率。'),
        CODE("review=R/'gemini_review/review.md'\ndisplay(Markdown(review.read_text(encoding='utf-8') if review.exists() else 'Gemini 評審尚未完成。'))"),
        MD('## 可重算驗證'),CODE("import subprocess, sys\nsubprocess.run([sys.executable,'verify_results.py'],check=True)\ndisplay(read('verification.json'))")])


if __name__=='__main__': main()
