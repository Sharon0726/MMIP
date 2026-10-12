"""Map the preserved Gemini blind review back to the two trained decoders."""
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results/gemini_review'


def main():
    raw = json.loads((R / 'raw_rows.json').read_text(encoding='utf-8'))
    mapping = json.loads((R / 'candidate_mapping.json').read_text(encoding='utf-8'))
    cases = {str(c['case']): c for c in json.loads((ROOT / 'results/gemini_cases.json').read_text(encoding='utf-8'))}
    assert raw[0] == ['case', 'candidate', 'verdict', 'reason']
    assert len(raw) == 13
    assert {(r[0], r[1]) for r in raw[1:]} == {(str(i), c) for i in range(1, 7) for c in ['A', 'B']}
    rows = []
    for case, candidate, verdict, reason in raw[1:]:
        assert verdict in ['符合', '部分符合', '不符合']
        model = mapping[case][candidate]
        # Only remove the trailing UI attachment-citation label, never alter the judgment.
        reason = re.sub(r'\s+JPG\s*$', '', reason).strip()
        rows.append(dict(case=int(case), candidate=candidate, model=model,
                         caption=cases[case][model], verdict=verdict, reason=reason))
    df = pd.DataFrame(rows)
    df.to_csv(R / 'review.csv', index=False, encoding='utf-8')
    counts = pd.crosstab(df.model, df.verdict).reindex(columns=['符合', '部分符合', '不符合'], fill_value=0)
    review = '''評審日期：2026-10-12。使用 **Google Gemini 網頁版 Flash 模式**；介面未顯示底層模型版本，因此不另推定版本。已實際上傳固定 test 前六張圖片與兩個模型的生成句，沒有提供參考描述。奇數案例 A=GRU、B=LSTM；偶數案例相反，提示中不揭露模型身分。

下表判定及理由保留 Gemini 原意，只移除網頁的 `JPG` 引用按鈕文字；另外加上對照模型與原始 caption。未修改任何不符合結果。

'''
    review += df.to_markdown(index=False) + '\n\n抽樣判定筆數（每模型 6 筆）：\n\n'
    review += counts.reset_index().to_markdown(index=False)
    review += '''

GRU 在這六張中有兩筆「符合」，LSTM 沒有完整符合案例；兩者各有一筆「部分符合」。這與本次 GRU 的 BLEU-4 較高方向一致，但 **六張質性抽樣不能估計全測試集準確率，也不足以證明架構優劣**。兩個模型都有把棒球、淋浴等核心事件說錯的案例，顯示能生成流暢句子不代表影像語意正確。

**評審歧義與限制：** 案例 4 的圖片是 rugby。Gemini 將 GRU 的 `football` 視為涵蓋 rugby 而判「符合」；若把 `football` 狹義解讀為美式足球或 association football，評價可能較低。因此保留原判定，同時指出此處有語詞歧義。模型與 Gemini 的細節判斷都可能出錯，不能以評審文字取代圖片證據。

查核檔案位於 `results/gemini_review/`：`prompt.txt` 為送出的提示；`raw_response.txt` 為可見回覆的原始文字（含網頁引用標籤）；`raw_rows.json` 保存原始表格欄位；`review.csv` 是 A/B 映射後的表格；`metadata.json` 記錄模式及原對話；`gemini_review_start.png` 與 `gemini_review.png` 保存網頁畫面。六張原圖與生成描述見上方例圖及 `results/gemini_cases.json`。
'''
    (R / 'review.md').write_text(review, encoding='utf-8')
    print(counts.to_string())
    print('Verified 12 unique candidate judgments and preserved original wording.')


if __name__ == '__main__':
    main()
