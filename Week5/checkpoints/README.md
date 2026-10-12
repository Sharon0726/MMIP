# 訓練權重

本機與 Colab 匯出壓縮檔保留 8 個訓練模型及 2 個 gallery 檔；大型 `.pt` 不加入 Git。

- `retrieval_resnet.pt`、`retrieval_vit.pt`：Quiz 1 已微調的影像 embedding 模型。
- `gallery_resnet.pt`、`gallery_vit.pt`：10,000 張資料庫的 embedding、原始 ID 與類別。
- `multimodal_resnet_fusion.pt`、`multimodal_vit_fusion.pt`：Quiz 2 圖文投影與融合分類器。
- `multimodal_resnet_image.pt`、`multimodal_resnet_text.pt`：單模態對照分類器。
- `caption_clip_gru.pt`、`caption_clip_lstm.pt`：Quiz 3 的影像條件投影與文字 decoder。

首次執行 `predict.py` 的多模態／caption 推論會由官方模型庫下載固定 backbone；retrieval 權重已包含其 backbone。從 Colab 匯出 `MMIP_Week5_results.zip` 後，把 `checkpoints/` 解壓到 Week5 根目錄即可使用。完整重現也可直接執行 Colab Runner。

```bash
python predict.py retrieval your_image.jpg --backbone vit
python predict.py multimodal your_image.jpg --text "a child is playing outside" --backbone resnet
python predict.py caption your_image.jpg --decoder gru
```
