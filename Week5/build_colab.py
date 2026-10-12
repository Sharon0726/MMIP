"""Build a clean Colab runner with embedded code and immutable Week4 baseline downloads."""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
SOURCES=['common.py','data_utils.py','models.py','caption_decoder.py','features.py','quiz1_retrieval.py',
         'quiz2_multimodal.py','quiz3_clip_caption.py','run_all.py','analyze_results.py','verify_results.py','test_invariants.py','predict.py','config.json','requirements.txt']
COMMIT='fe4592d6dd33388d3486c227a63eefc157843e88'


def main():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES: z.writestr(name,(ROOT/name).read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    manifest={}
    for p in (ROOT/'baseline_week4').iterdir():
        if p.suffix not in ['.csv','.json']: continue
        folder='predictions/' if p.name in ['caption_gru.json','caption_lstm.json'] else ''
        manifest[p.name]={'url':f'https://raw.githubusercontent.com/Sharon0726/MMIP/{COMMIT}/Week4/results/{folder}{p.name}',
                          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    bootstrap='''import base64, hashlib, io, os, pathlib, subprocess, sys, urllib.request, zipfile, torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU.'
print('GPU:',torch.cuda.get_device_name(0))
work=pathlib.Path('/content/mmip_week5'); work.mkdir(exist_ok=True)
(work/'results').mkdir(exist_ok=True); (work/'baseline_week4').mkdir(exist_ok=True)
payload='''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z: z.extractall(work)
manifest='''+repr(manifest)+'''
for name, item in manifest.items():
    data=urllib.request.urlopen(item['url'],timeout=60).read()
    assert hashlib.sha256(data).hexdigest()==item['sha256'], name
    (work/'baseline_week4'/name).write_bytes(data)
os.chdir(work)
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r','requirements.txt'])
subprocess.check_call([sys.executable,'-m','unittest','test_invariants','-v'])
'''
    runner='''process=subprocess.Popen([sys.executable,'-u','run_all.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
with open('results/training.log','w') as log:
    for line in process.stdout:
        print(line,end='',flush=True); log.write(line); log.flush()
assert process.wait()==0, 'Training failed; inspect the log.'
'''
    export='''from google.colab import files
import pandas as pd
for name in ['retrieval_metrics','multimodal_metrics','caption_comparison']:
    display(pd.read_csv('results/'+name+'.csv'))
out=pathlib.Path('/content/MMIP_Week5_results.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['results','checkpoints']:
        for p in pathlib.Path(folder).rglob('*'):
            if p.is_file(): z.write(p,p.as_posix())
files.download(str(out))
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# MMIP Week 5：多模態深度學習\n\n蘇芯儀 / Sharon0726\n\nT4 GPU，依序執行。三題含進階：CNN/ViT 檢索、圖文融合分類、CLIP Captioning 與 Week4 BLEU/BERTScore 比較。程式內嵌；Week4 基準由固定 commit 下載並檢查雜湊。重跑會重寫同名結果，請先匯出需要保留的實驗。'),
        nbf.v4.new_code_cell(bootstrap,metadata={'cellView':'form'}),nbf.v4.new_code_cell(runner),nbf.v4.new_code_cell(export)],
        metadata={'colab':{'name':'MMIP_Week5_Colab.ipynb'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}})
    nbf.validate(nb); nbf.write(nb,ROOT/'MMIP_Week5_Colab.ipynb')
    (ROOT/'.local').mkdir(exist_ok=True)
    (ROOT/'.local/baseline_download_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Created clean Colab runner with',len(SOURCES),'embedded files and',len(manifest),'pinned baseline files')


if __name__=='__main__': main()
