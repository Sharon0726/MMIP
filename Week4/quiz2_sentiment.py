"""Train RNN and LSTM on the same movie-review splits; test only selected checkpoints."""
import time
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common import CFG, DEVICE, ROOT, classification_metrics, encode, fingerprint, make_vocab, save_json, save_predictions, seed_all, sync
from models import SentimentModel


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    probs, labels, loss_sum = [], [], 0.
    for x, length, y in loader:
        logits = model(x.to(DEVICE), length.to(DEVICE))
        loss_sum += nn.functional.cross_entropy(logits, y.to(DEVICE), reduction='sum').item()
        probs.append(logits.softmax(-1).cpu().numpy())
        labels.extend(y.tolist())
    return np.array(labels), np.concatenate(probs), loss_sum / len(labels)


def main():
    c = CFG['sentiment']
    seed_all(CFG['seed'])
    ds = load_dataset('cornell-movie-review-data/rotten_tomatoes')
    # Remove exact normalized text overlaps, preserving the official test first.
    seen, records, splits = set(), [], {}
    for split in ['test', 'validation', 'train']:
        keep = []
        for i, row in enumerate(ds[split]):
            key = fingerprint(' '.join(row['text'].lower().split()))
            if key not in seen:
                seen.add(key)
                keep.append(i)
                records.append({'split': split, 'source_id': i, 'label': row['label'], 'sha256': key})
        splits[split] = ds[split].select(keep)
    pd.DataFrame(records).to_csv(ROOT / 'results/sentiment_split.csv', index=False)
    vocab = make_vocab(splits['train']['text'], c['vocab_size'])
    save_json('results/sentiment_vocab.json', vocab)
    tensors = {}
    for split, data in splits.items():
        encoded = [encode(t, vocab, c['max_length']) for t in data['text']]
        tensors[split] = TensorDataset(torch.tensor([x[0] for x in encoded]), torch.tensor([x[1] for x in encoded]), torch.tensor(data['label']))
    save_json('results/sentiment_data.json', {'source': 'cornell-movie-review-data/rotten_tomatoes',
        'fingerprints': {k: ds[k]._fingerprint for k in ds}, 'counts': {k: len(v) for k,v in splits.items()},
        'vocab_size': len(vocab), 'labels': {'0': 'negative', '1': 'positive'},
        'deduplication': 'normalized exact text; test > validation > train'})
    metrics = []
    for kind in ['rnn', 'lstm']:
        seed_all(CFG['seed'])
        loaders = {s: DataLoader(t, batch_size=c['batch_size'], shuffle=s=='train') for s,t in tensors.items()}
        model = SentimentModel(len(vocab), kind, c['embedding'], c['hidden']).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
        best, best_epoch, history = -1, None, []
        sync(); start = time.perf_counter()
        for epoch in range(1, c['epochs'] + 1):
            model.train(); loss_sum = 0.
            for x, length, y in loaders['train']:
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x.to(DEVICE), length.to(DEVICE)), y.to(DEVICE))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                loss_sum += loss.item() * len(y)
            y, p, val_loss = evaluate(model, loaders['validation'])
            score = classification_metrics(y, p)['macro_f1']
            history.append({'epoch': epoch, 'train_loss': loss_sum/len(tensors['train']), 'val_loss': val_loss, 'val_macro_f1': score})
            print(f'{kind} epoch {epoch}: val_F1={score:.4f}', flush=True)
            if score > best:
                best, best_epoch = score, epoch
                torch.save(model.state_dict(), ROOT / f'checkpoints/sentiment_{kind}.pt')
        sync(); elapsed = time.perf_counter() - start
        pd.DataFrame(history).to_csv(ROOT / f'results/history/sentiment_{kind}.csv', index=False)
        model.load_state_dict(torch.load(ROOT / f'checkpoints/sentiment_{kind}.pt', map_location=DEVICE, weights_only=True))
        y, p, _ = evaluate(model, loaders['test'])
        save_predictions(f'sentiment_{kind}', range(len(y)), y, p, splits['test']['text'])
        row = {'model': kind, 'best_epoch': best_epoch, 'train_seconds': elapsed,
               'parameters': sum(p.numel() for p in model.parameters()), **classification_metrics(y,p)}
        metrics.append(row); print(row, flush=True)
    pd.DataFrame(metrics).to_csv(ROOT / 'results/sentiment_metrics.csv', index=False)


if __name__ == '__main__':
    main()
