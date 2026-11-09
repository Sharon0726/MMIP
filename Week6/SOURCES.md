# 資料、模型與方法來源

- 作業要求：老師2026-10-21「多模態語言模型如何運作」簡報第64–66頁。使用者提供的原檔位於本機Week6；本repo僅整理要求及實驗，未重新散布整份講義。
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)：checkpoint `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。模型卡標示Apache-2.0。
- [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)：checkpoint `482adb537c021c86670beed01cd58990d01e72e4`。模型卡標示Apache-2.0。
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)：checkpoint `66285546d2b821cf421d4f5eb2576359d3770cd3`；僅用於B的推論，未散布其基底權重。
- [TRL 0.23.1 DPOTrainer](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer)：文字與視覺偏好訓練使用此版本。
- [Transformers bitsandbytes量化](https://huggingface.co/docs/transformers/quantization/bitsandbytes)：INT8與NF4設定。
- [scikit-image data](https://scikit-image.org/docs/stable/api/skimage.data.html)：B01 astronaut、B02 chelsea、B03 coffee、B04 rocket；來源逐題寫在B_benchmark.json。astronaut來自NASA；其餘照片來源與授權說明見各data函式文件。這些範例影像的權利仍屬原作者。
- B05–B20與全部視覺偏好圖使用PIL程式生成；OCR使用DejaVu Sans字型。文字偏好為虛構訂單，依規則產生chosen/rejected；沒有真人回饋或個資資料集。
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)：GPU與使用額度會變動。本次商用judge由Colab內建 `google.colab.ai.generate_text` 呼叫，模型名為 `google/gemini-3.5-flash-lite`。

`C_text_adapter`與`V_adapter`只含本次學到的LoRA權重和對應tokenizer／processor檔案，需搭配上述基底模型；不包含完整基底權重。Qwen附其上游LICENSE；SmolVLM2該revision沒有獨立LICENSE檔，依模型卡標示附上Apache官方2.0授權文本。adapter README為2026-11-09補上的模型說明，科學數據與safetensors未更改。

此資料夾沒有加入帳號憑證、講義完整PDF、瀏覽器截圖或私人Colab工作階段metadata。原始Colab科學輸出與套件警告保留於已執行Notebook。
