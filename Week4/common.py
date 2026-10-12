"""Shared reproducibility, tokenization, metrics and result helpers."""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for folder in ['results', 'results/figures', 'results/predictions', 'results/history', 'checkpoints', 'cache']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def save_json(path, obj):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text.lower())


def make_vocab(texts, limit):
    counter = Counter(t for text in texts for t in tokens(text))
    words = sorted(counter, key=lambda t: (-counter[t], t))[:limit - 4]
    return {w: i for i, w in enumerate(['<pad>', '<unk>', '<bos>', '<eos>'] + words)}


def encode(text, vocab, length, caption=False):
    ids = [vocab.get(t, 1) for t in tokens(text)]
    ids = [2] + ids[:length - 2] + [3] if caption else (ids[:length] or [1])
    n = len(ids)
    return ids + [0] * (length - n), n


def classification_metrics(y, p):
    pred = p.argmax(1)
    out = {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, average='macro'),
           'macro_precision': precision_score(y, pred, average='macro', zero_division=0),
           'macro_recall': recall_score(y, pred, average='macro', zero_division=0)}
    out['auc'] = roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class='ovr', average='macro')
    return {k: float(v) for k, v in out.items()}


def save_predictions(name, ids, y, p, texts=None):
    df = pd.DataFrame({'id': ids, 'true': y, 'predicted': p.argmax(1)})
    for i in range(p.shape[1]):
        df[f'p_{i}'] = p[:, i]
    if texts is not None:
        df['text'] = texts
    df.to_csv(ROOT / f'results/predictions/{name}.csv', index=False)


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()
