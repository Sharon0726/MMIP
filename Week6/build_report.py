"""Build the GitHub-readable report from saved, verified CSV/JSON evidence."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'artifacts'


def table(name, columns):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    lines = ['| ' + ' | '.join(label for _,label in columns) + ' |',
             '|' + '|'.join('---' for _ in columns) + '|']
    for row in rows:
        vals = []
        for key,_ in columns:
            val = row[key]
            try:
                number = float(val)
                val = f'{number:.3f}' if number != int(number) else str(int(number))
            except ValueError:
                pass
            vals.append(val.replace('|','\\|'))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


text = '''# MMIP Week6 實驗報告

本報告對應2026-10-21老師簡報第64–66頁。GPU實驗日期為2026-11-04，2026-11-09重新核對全部要求與保存資料。[15項要求對照](README.md#老師完整要求與對應證據)與[離線查核結果](verification.json)提供證據入口。

## 實驗設計

全部模型推論與訓練在Colab Tesla T4執行。A使用Qwen2.5-1.5B-Instruct；B使用SmolVLM2-2.2B-Instruct及Qwen2.5-VL-3B-Instruct；C在前兩者中的文字模型／SmolVLM2上做DPO。固定checkpoint SHA、seed 20261104、greedy decoding、batch size 1。原生BF16 checkpoint轉為FP16作未量化基準，4-bit使用NF4與double quantization，運算精度FP16。

A有20題正式測試、10題獨立開發集；B有20題。文字偏好資料120訓練／20驗證／30目標測試，另外100題通用測試；視覺偏好資料120訓練／20驗證／20測試。驗證集觀察訓練，固定跑2 epochs，未使用測試集挑checkpoint。

商用judge為Colab內建Gemini 3.5 Flash Lite。每8個匿名答案一批，按0–4分規準評分；完整正確4、輕微格式問題3、部分正確2、多數錯誤1、錯誤／空白0。相同題目、參考、答案共用快取，保留全部請求與回覆。商用評分可能偏誤，故同時報告客觀代理指標。[原始judge紀錄](artifacts/commercial_judge_raw.json)

## A：提示與量化

20題涵蓋算術、邏輯、閱讀、翻譯、格式，每類4題。數字採數值比對，格式題驗JSON或嚴格文字，翻譯採關鍵語意覆蓋率。因此「客觀平均分」含部分得分，不應全部稱為完全正確率。

baseline是一般指令；few-shot提供獨立例題；CoT要求最多三個簡短步驟。開發集baseline／few-shot／CoT分別0.8／1.0／0.6，依預先規則在few-shot與CoT間選few-shot，再做4-bit組合實驗。

'''
text += table('A_summary.csv',[('condition','條件'),('n','題數'),('objective','客觀平均分'),('judge_0_to_4','商用平均／4'),('mean_seconds','秒／題'),('truncated','截斷')])
text += '''

few-shot在開發集較好，正式測試FP16卻由0.70降到0.55，顯示小型開發集的提示選擇未泛化。4-bit baseline只有0.25，加few-shot回到0.55，但仍低於FP16 baseline。INT8客觀分保持0.70，商用分略降且速度較慢，不能宣稱量化必然更快。

### 顯存

原始NVML每0.1秒取樣整張GPU使用量，包含CUDA context、配置快取及其他使用量；平均以生成期間時間積分計算，不含載入模型。原始跨精度數值很接近，因此另以每條件全新Python程序量測該程序的NVML顯存。同樣暖機、同20題；100個新答案全部與原答案一致，未用重測結果重新挑提示。

'''
text += table('A_fresh_process_memory_summary.csv',[('condition','條件'),('process_gpu_mean_mib','程序平均MiB'),('process_gpu_peak_mib','程序峰值MiB'),('model_footprint_mib','模型占用MiB'),('answers_equal_original','與原答案一致／20')])
text += '''

FP16 baseline平均3167.852 MiB，4-bit＋few-shot為1654 MiB，下降47.8%。這是程序顯存，不是整張GPU或純權重記憶體。與FP16相比的節省伴隨客觀平均分0.70→0.55的代價。

![顯存](artifacts/A_memory_verified.png)

## B：兩個VLM

辨識、計數、空間關係、OCR、描述各4題；4張scikit-image公開範例照片與16張PIL自製控制圖。兩模型對同一圖片使用完全相同英文問題；各自processor與影像token數不同，因此速度／顯存差異不能僅歸因於參數數量。

![20張圖片](artifacts/B_benchmark_contact_sheet.png)

'''
text += table('B_final_summary.csv',[('condition','模型'),('n','題數'),('objective','客觀平均分'),('judge_0_to_4','商用平均／4'),('mean_seconds','秒／題'),('gpu_mean_mib','整張GPU平均MiB')])
text += '''

SmolVLM2的辨識與形狀位置均正確，但計數與OCR各只對1/4。例：把5個藍色方塊回答成7；「MMIP 2026」讀成「MMMP 2026」；兩個紅色方塊描述成三個。Qwen在這20題表現較佳，但測試以簡單控制圖為主，不代表全面能力排名。

評分稽核一致修正兩模型的規則：問題只問形狀，不能因未寫顏色扣分；左右反向的等價敘述應接受；參考答案補上白色背景這個真實細節。修正前紀錄保留，最終使用B_final_*與B_v2_judged.json。客觀描述分數只是概念覆蓋代理，商用評分補足格式及錯誤資訊檢查。

Gemini此處接收題目、可信的場景標註及模型回答，**未直接接收圖片**，因此是參考答案式的商用LLM評估，不能宣稱獨立視覺驗證。

## C：文字DPO

目標為訂單資料擷取：輸出buyer、item、quantity的JSON，沒提供的欄位必須為null。chosen忠於證據；rejected填入不存在的資訊或錯誤數量。偏好是程式規則合成，非真人回饋調查。測試使用不同顧客代碼與新模板；通用100題為算術、邏輯、英文、科學常識、大小比較各20題。

QLoRA：4-bit基底；LoRA r=8、alpha=16、dropout=0.05，q_proj/v_proj；DPO beta=0.1、學習率5e-5、2 epochs、microbatch 1、累積4次梯度。共60次optimizer更新；文字可訓練參數1,089,536。前後皆使用相同4-bit基底與提示，不將量化差異算成微調效果。

'''
text += table('C_final_summary.csv',[('phase','階段'),('target_n','目標題數'),('target_exact','JSON完全正確率'),('valid_json','合法JSON率'),('missing_field_error_rate','缺失欄位錯誤率'),('invented_field_rate','捏造欄位率'),('general_n','通用題數'),('general_accuracy','通用正確率')])
text += '''

30題目標測試微調前後均100%，有天花板效應，不能證明微調提升。通用61%→60%僅差一題，也不能斷言嚴重災難性遺忘。訓練首／末記錄loss為0.6907／0.1875，證明偏好目標下降，並不代表測試能力提升。

missing_field_error_rate包含缺鍵、應為null卻填值、無效JSON的失敗；invented_field_rate只計可解析JSON中把應為null的欄位填上值。兩者分開，避免把所有格式錯誤都叫做幻覺。訓練模板與缺漏模式有相關，未來應增加多樣自然語句、缺漏組合與更難的獨立測試。

## C進階：視覺DPO及擾動

目標是避免列出圖中不存在的circle、square、triangle、star。每張圖含1–3種形狀；chosen列真實形狀，rejected多加一個不存在的形狀。160張圖在訓練／驗證／測試間無相同檔案hash。使用4-bit SmolVLM2、DPO學習率2e-5，其他LoRA和batch設定同文字實驗；2 epochs、60步、2,568,192個可訓練參數。

同20張測試圖分為原圖、亮度乘0.35、縮至32×32後JPEG quality20並放回原大小。前後各60次推論，生成上限同為32 tokens。

![擾動圖片](artifacts/V_perturbation_examples.png)

'''
text += table('V_summary.csv',[('phase','階段'),('condition','影像條件'),('n','圖數'),('shape_f1','F1'),('shape_recall','召回率'),('hallucination_rate','不存在形狀提及率'),('exact_set','集合完全正確率'),('truncated','截斷數')])
text += '''

原圖F1由0.687→0.706，幻覺率0.429→0.404，僅小幅改善；變暗圖F1由0.698→0.692，並未改善。低畫質F1由0.664→0.671。訓練首／末loss為0.6933／0.6278，仍不足以主張全面提升。

Precision衡量提及形狀有多少是真的，recall衡量實際形狀找回多少，F1是兩者調和平均。幻覺率為提及集合中不存在形狀所占比例，重複提及合併為集合。Recall一直為1，顯示主要問題是多說不存在的形狀；同時報F1才能避免「什麼都不說」的模型看似無幻覺。此評分不衡量完整句子品質，亦受截斷預算限制；不能推廣到自然照片的全面幻覺能力。

## 總覽與查核

![結果總覽；商用分數除以4以便同軸呈現](artifacts/final_results_dashboard.png)

文字、視覺實際訓練分別172.7、173.1秒，不含下載與前後測試。兩個adapter的所有tensor均為有限值，LoRA B矩陣均有非零更新；對應資料與訓練紀錄保留。`verify_results.py`重算客觀分數、平均顯存及結果表，也檢查權重；沒有再次GPU訓練或重呼叫judge。

完整流程：定義能力 → 建立題庫與標準 → 分開開發／訓練／驗證／測試 → 固定其他條件比較 → 同看品質與資源 → 留下原始紀錄 → 分析限制。測試小、僅一seed，沒有統計顯著性結論。實驗結果不必全部變好，必須忠於證據。

## 名詞速查

| 名詞 | 在本作業的意思 |
|---|---|
| LLM / VLM | 文字模型／把圖片與文字一起處理的模型 |
| Inference / Fine-tuning | 用既有權重回答／實際更新參數 |
| Benchmark | 固定考題及評分方法 |
| Few-shot / CoT | 提供少量例題／要求簡短解題步驟；不直接改權重 |
| Token | 模型處理的單位，不等於一個中文字或單字 |
| Quantization | 降低權重精度；本次比較FP16、INT8、NF4 |
| VRAM | GPU記憶體；含權重、KV cache、中間值與context等 |
| LoRA / QLoRA | 訓練少量低秩參數／在量化基底上做LoRA |
| Preference / DPO | 同題好壞答案配對／直接學相對偏好，無需另建reward model跑PPO |
| Epoch / Gradient accumulation | 看完一輪資料／累積數個microbatch再更新 |
| Hallucination | 捏造輸入中不存在的資訊 |
| Precision / Recall / F1 | 說出的有多少正確／真的找回多少／兩者調和平均 |
| Catastrophic forgetting | 微調新任務後損失原有能力，因此另測通用題 |
| LLM-as-a-judge | 商用模型依規則評答案；保留理由並配合客觀指標 |

模型、照片及方法連結：[SOURCES.md](SOURCES.md)。逐題資料與完整程式：[README.md](README.md)。
'''
(ROOT/'REPORT.md').write_text(text,encoding='utf-8')
print('Built REPORT.md from saved evidence.')
