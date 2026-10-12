"""Pinned datasets and image-level partitions shared across experiments."""
from functools import lru_cache
import hashlib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torchvision import transforms
from common import ROOT, CFG, save_json


@lru_cache(maxsize=1)
def cifar():
    name = 'uoft-cs/cifar10'
    return load_dataset(name, revision=CFG['revisions'][name])


@lru_cache(maxsize=1)
def flickr():
    name = 'tsystems/flickr8k'
    return load_dataset(name, revision=CFG['revisions'][name], split='train')


def image_transform(kind, train=False):
    mean, std = ((0.5,)*3, (0.5,)*3) if kind == 'vit' else ((.485,.456,.406),(.229,.224,.225))
    ops = [transforms.Resize((224,224))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(mean,std)])


class ImageRows(torch.utils.data.Dataset):
    def __init__(self, data, ids, transform, image_key='img', label_key='label'):
        self.data, self.ids, self.transform = data, list(ids), transform
        self.image_key, self.label_key = image_key, label_key
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, j):
        row = self.data[self.ids[j]]
        return self.transform(row[self.image_key].convert('RGB')), row[self.label_key] if self.label_key else self.ids[j]


def flickr_partitions():
    ds = flickr()
    df = pd.read_csv(ROOT/'baseline_week4/caption_split.csv')
    assert len(df)==len(ds) and df.source_id.is_unique and df.filename.is_unique
    names = ds['image_filename']
    for row in df.itertuples():
        assert names[row.source_id] == row.filename, 'Dataset order changed from Week4.'
    # Preserve the saved Week4 order, not a freshly generated split.
    ids = {s:df.loc[df.split==s,'source_id'].tolist() for s in ['train','validation','test']}
    assert not set(ids['train']) & set(ids['test'])
    df.to_csv(ROOT/'results/flickr_split.csv',index=False)
    save_json('results/flickr_data.json', {'dataset':'tsystems/flickr8k','revision':CFG['revisions']['tsystems/flickr8k'],
        'fingerprint':ds._fingerprint,'counts':{s:len(v) for s,v in ids.items()},
        'split':'Exact Week4 custom image-level split; first caption only for binary pairing; all five for caption training.'})
    return ds, ids
