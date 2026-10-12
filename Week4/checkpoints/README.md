# 訓練權重

本機及 Colab 結果壓縮檔保留六個實際訓練 checkpoint：

- `sentiment_rnn.pt`、`sentiment_lstm.pt`
- `vision_vit.pt`、`vision_resnet.pt`
- `caption_gru.pt`、`caption_lstm.pt`

大型二進位權重不提交 GitHub。可由 `MMIP_Week4_Colab.ipynb` 重新訓練及匯出。結果、模型參數、逐筆預測、來源程式與 checkpoint SHA-256/大小保留供查核。

推論範例（位於 Week4，且此資料夾已有權重）：

```bash
python predict.py sentiment "A thoughtful and moving film." --model lstm
python predict.py vision path/to/image.jpg --model vit
python predict.py caption path/to/photo.jpg --model lstm
```
