"""Record artifact integrity after downloading the real Colab results."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=json.loads((ROOT/'results/source_sha256.json').read_text())
    for name,sha in source.items():
        assert digest(ROOT/name)==sha, f'Training source mismatch: {name}'
    checkpoints={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in (ROOT/'checkpoints').glob('*.pt')}
    assert len(checkpoints)==6
    (ROOT/'results/checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2))
    env=json.loads((ROOT/'results/environment.json').read_text())
    (ROOT/'requirements-colab-record.txt').write_text('# Exact observed package versions; install torch/torchvision for your platform.\n'+'\n'.join(f'{k}=={v}' for k,v in env['packages'].items())+'\n')
    (ROOT/'results/completed.json').write_text(json.dumps({'training_complete':True,'source_hashes_match':True,'packaged_utc':datetime.now(timezone.utc).isoformat()},indent=2))
    print('Verified training source hashes and recorded six checkpoint hashes.')


if __name__=='__main__': main()
