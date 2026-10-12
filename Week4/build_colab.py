"""Build a self-contained Colab notebook from the audited local sources."""
import base64
import io
import zipfile
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
SOURCES=['common.py','models.py','quiz2_sentiment.py','quiz3_vision.py','quiz4_caption.py','run_all.py','analyze_results.py','verify_results.py','predict.py','config.json','requirements.txt']


def main():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES: z.writestr(name,(ROOT/name).read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    bootstrap='''import base64, io, os, pathlib, subprocess, sys, zipfile, torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU.'
print('GPU:', torch.cuda.get_device_name(0))
work=pathlib.Path('/content/mmip_week4'); work.mkdir(exist_ok=True)
(work/'results').mkdir(exist_ok=True)
payload='''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z: z.extractall(work)
os.chdir(work)
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r','requirements.txt'])
'''
    runner='''process=subprocess.Popen([sys.executable,'-u','run_all.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
with open('results/training.log','w') as log:
    for line in process.stdout:
        print(line,end='',flush=True); log.write(line); log.flush()
assert process.wait()==0, 'Training failed; inspect the log.'
'''
    export='''from google.colab import files
out=pathlib.Path('/content/MMIP_Week4_results.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['results','checkpoints']:
        for p in pathlib.Path(folder).rglob('*'):
            if p.is_file(): z.write(p,p.as_posix())
print('Export bytes:',out.stat().st_size)
files.download(str(out))
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# MMIP Week 4 — 四題 GPU 實驗\n\n蘇芯儀 / Sharon0726\n\nT4 GPU，依序執行。Rotten Tomatoes RNN/LSTM；CIFAR-10 ViT/ResNet；Flickr8k GRU/LSTM Captioning。包含真實訓練、測試、BLEU、計時與驗證。Gemini 人工評審另外見 REPORT.md。'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell(runner),nbf.v4.new_code_cell(export)],metadata={'colab':{'name':'MMIP_Week4_Colab.ipynb'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}})
    nbf.write(nb,ROOT/'MMIP_Week4_Colab.ipynb')
    (ROOT/'.local').mkdir(exist_ok=True)
    (ROOT/'.local/bootstrap.txt').write_text(bootstrap+'\n'+runner+'\n'+export,encoding='utf-8')
    print('Created self-contained notebook')


if __name__=='__main__': main()
