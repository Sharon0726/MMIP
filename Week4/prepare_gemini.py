"""Prepare reproducible blind A/B prompts for Gemini web image review."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    cases=json.loads((ROOT/'results/gemini_cases.json').read_text(encoding='utf-8'))
    directory=ROOT/'results/gemini_review'; directory.mkdir(exist_ok=True)
    prompt='''你是影像描述的獨立評審。這是學術作業的六張 Flickr8k 測試圖片，檔名 caption_sample_1.jpg 到 caption_sample_6.jpg 對應案例 1 到 6。每張圖片有兩個模型實際生成的描述 A/B；請依圖片本身判斷，不要根據模型身分、句子流暢度或猜測參考答案評分。

逐一評估每個 A/B 描述的主要物體、數量、動作、屬性和空間關係。評為「符合」：重要內容皆可由圖片支持，省略細節可接受；「部分符合」：主要物體或場景合理，但有錯誤動作、數量或其他重要細節；「不符合」：主要物體/事件不符，或幻覺是核心內容。請明確寫出圖片能支持及不能支持的部分；無法辨識則說不確定，不要補出細節。

請以繁體中文回覆一個 Markdown 表格，共 12 列，欄位為 case、candidate、verdict、reason。每個 reason 限一到兩句。最後給兩到三句整體限制說明。不要提供新 caption，不要修改原始候選句。
'''
    mapping={}
    for case in cases:
        i=case['case']; a,b=('gru','lstm') if i%2 else ('lstm','gru')
        mapping[str(i)]={'A':a,'B':b}
        prompt+=f'\n案例 {i}（caption_sample_{i}.jpg）\nA: {case[a]}\nB: {case[b]}\n'
    (directory/'prompt.txt').write_text(prompt,encoding='utf-8')
    (directory/'candidate_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print('Prepared prompt and alternating A/B model mapping for the fixed first six test images.')


if __name__=='__main__': main()
