"""Run real GPU experiments and preserve their environment and source identity."""
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime,timezone
import torch
from common import ROOT,CFG,DEVICE,save_json


if __name__=='__main__':
    save_json('results/environment.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
        'platform':platform.platform(),'device':str(DEVICE),'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
        'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','datasets','timm','transformers','bert-score','nltk','numpy','pandas','scikit-learn','Pillow']}})
    save_json('results/config.json',CFG)
    save_json('results/source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    save_json('results/baseline_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'baseline_week4').iterdir() if p.is_file()})
    for name in ['quiz1_retrieval.py','quiz2_multimodal.py','quiz3_clip_caption.py','analyze_results.py','verify_results.py']:
        print(f'Running {name}',flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/name)],cwd=ROOT,check=True)
    save_json('results/completed.json',{'training_complete':True,'completed_utc':datetime.now(timezone.utc).isoformat()})
    print('ALL_WEEK5_EXPERIMENTS_COMPLETE',flush=True)
