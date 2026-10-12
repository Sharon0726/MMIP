"""Run the three real training pipelines and capture complete provenance."""
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from common import ROOT, CFG, DEVICE, save_json
import torch

if __name__=='__main__':
    save_json('results/environment.json', {'started_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'platform':platform.platform(),'device':str(DEVICE),
        'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
        'packages':{k:importlib.metadata.version(k) for k in ['torch','torchvision','timm','datasets','numpy','pandas','scikit-learn','nltk','Pillow']}})
    save_json('results/config.json',CFG)
    save_json('results/source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    for script in ['quiz2_sentiment.py','quiz3_vision.py','quiz4_caption.py','analyze_results.py','verify_results.py']:
        print(f'\n=== {script} ===',flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)
    save_json('results/completed.json', {'completed_utc':datetime.now(timezone.utc).isoformat(),'training_complete':True})
    print('ALL_EXPERIMENTS_COMPLETE',flush=True)
