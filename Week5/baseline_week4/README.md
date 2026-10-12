# Week4 基準來源

五個 CSV / JSON 檔原樣複製自 [MMIP Week4 固定 commit](https://github.com/Sharon0726/MMIP/tree/fe4592d6dd33388d3486c227a63eefc157843e88/Week4/results)。`caption_gru.json`、`caption_lstm.json` 原位於 `results/predictions/`。

- `caption_split.csv`：Flickr8k 6000 train / 1000 validation / 1091 test，按影像切分。
- `caption_vocab.json`：只使用 Week4 train captions 建立的字典。
- `caption_gru.json`、`caption_lstm.json`：ResNet18 encoder + GRU / LSTM 的 1091 張 test 預測，每張包含五句參考。
- `caption_metrics.csv`：Week4 原始 BLEU 與訓練紀錄。

Week5 直接用既有輸出重新計算一致設定的 BLEU 與 BERTScore，未改寫生成描述。檔案 SHA-256 記錄在 `../results/baseline_sha256.json`。
