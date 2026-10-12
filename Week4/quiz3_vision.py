"""Fine-tune a pretrained ViT block / ResNet stage, then compare on identical test data."""
import time
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datasets import load_dataset
from common import CFG, DEVICE, ROOT, classification_metrics, save_json, save_predictions, seed_all, sync
from models import make_vision, vision_train_mode


@lru_cache(maxsize=1)
def cifar_data():
    return load_dataset('uoft-cs/cifar10')


class Cifar10(torch.utils.data.Dataset):
    """Official CIFAR-10 split mirrored by UofT on Hugging Face (faster CDN)."""
    def __init__(self, root=None, train=True, download=False, transform=None):
        self.data = cifar_data()['train' if train else 'test']
        self.targets = self.data['label']
        self.classes = self.data.features['label'].names
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        row = self.data[i]
        img = row['img'].convert('RGB')
        return (self.transform(img) if self.transform else img), row['label']


def image_transform(kind, train=False):
    # Match each pretrained model's normalization; augmentation and size are shared.
    mean, std = ((0.5,)*3, (0.5,)*3) if kind == 'vit' else ((.485,.456,.406),(.229,.224,.225))
    ops = [transforms.Resize((224,224))]
    if train:
        ops += [transforms.RandomHorizontalFlip()]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(mean,std)])


@torch.inference_mode()
def evaluate(model, loader):
    model.eval(); probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type=='cuda'):
            logits = model(x.to(DEVICE))
        probs.append(logits.float().softmax(-1).cpu().numpy()); ys.extend(y.tolist())
    return np.array(ys), np.concatenate(probs)


def main():
    c = CFG['vision']; seed_all(CFG['seed'])
    base = Cifar10(train=True)
    test_base = Cifar10(train=False)
    rng = np.random.default_rng(CFG['seed'])
    ids = {'train': [], 'validation': [], 'test': []}
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(np.array(base.targets)==label))
        ids['train'].extend(indices[:c['train_per_class']].tolist())
        ids['validation'].extend(indices[c['train_per_class']:c['train_per_class']+c['val_per_class']].tolist())
        test_indices = rng.permutation(np.flatnonzero(np.array(test_base.targets)==label))
        ids['test'].extend(test_indices[:c['test_per_class']].tolist())
    for split in ids:
        ids[split] = sorted(ids[split])
    assert not set(ids['train']) & set(ids['validation'])
    pd.DataFrame([{'split':s, 'source': 'official_test' if s=='test' else 'official_train',
                   'source_id':i, 'label':(test_base if s=='test' else base).targets[i]}
                  for s, indices in ids.items() for i in indices]).to_csv(ROOT/'results/vision_split.csv',index=False)
    save_json('results/vision_data.json', {'source':'uoft-cs/cifar10', 'fingerprints':{k:d._fingerprint for k,d in cifar_data().items()}, 'counts':{k:len(v) for k,v in ids.items()},
        'classes':base.classes, 'selection':'seeded class-balanced subset for train/val; full official test',
        'train_unused':len(base)-len(ids['train'])-len(ids['validation']),
        'pretraining':{'vit':'ImageNet-21k then ImageNet-1k', 'resnet':'ImageNet-1k'}})
    metrics=[]
    for kind in ['vit','resnet']:
        seed_all(CFG['seed'])
        loaders = {}
        for split in ids:
            data = Cifar10(train=split!='test',transform=image_transform(kind, split=='train'))
            loaders[split]=DataLoader(Subset(data,ids[split]),batch_size=c['batch_size'],shuffle=split=='train',num_workers=2,pin_memory=DEVICE.type=='cuda')
        model=make_vision(kind).to(DEVICE)
        # Save evidence that a pretrained backbone tensor actually changed.
        probe_name = 'blocks.11.attn.qkv.weight' if kind=='vit' else 'layer4.0.conv1.weight'
        before=dict(model.named_parameters())[probe_name].detach().cpu().clone()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=0.01)
        scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda')
        best=-1; history=[]; sync(); start=time.perf_counter()
        for epoch in range(1,c['epochs']+1):
            vision_train_mode(model,kind); loss_sum=0.
            for x,y in loaders['train']:
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type=='cuda'):
                    loss=nn.functional.cross_entropy(model(x.to(DEVICE)),y.to(DEVICE))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                loss_sum+=loss.item()*len(y)
            y,p=evaluate(model,loaders['validation']); score=classification_metrics(y,p)
            history.append({'epoch':epoch,'train_loss':loss_sum/len(ids['train']), 'val_accuracy':score['accuracy'],'val_macro_f1':score['macro_f1']})
            print(f'{kind} epoch {epoch}: val_acc={score["accuracy"]:.4f}',flush=True)
            if score['accuracy']>best:
                best=score['accuracy']; best_epoch=epoch
                torch.save(model.state_dict(),ROOT/f'checkpoints/vision_{kind}.pt')
        sync(); elapsed=time.perf_counter()-start
        model.load_state_dict(torch.load(ROOT/f'checkpoints/vision_{kind}.pt',map_location=DEVICE,weights_only=True))
        delta=(dict(model.named_parameters())[probe_name].detach().cpu()-before).abs().max().item()
        assert delta>0, 'Backbone did not change: not fine-tuning!'
        y,p=evaluate(model,loaders['test'])
        save_predictions(f'vision_{kind}',ids['test'],y,p)
        pd.DataFrame(history).to_csv(ROOT/f'results/history/vision_{kind}.csv',index=False)
        row={'model':kind,'best_epoch':best_epoch,'train_seconds':elapsed,
             'total_parameters':sum(p.numel() for p in model.parameters()),
             'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'backbone_probe':probe_name,'backbone_max_abs_change':delta,**classification_metrics(y,p)}
        metrics.append(row); print(row,flush=True)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(ROOT/'results/vision_metrics.csv',index=False)
    # Save deterministic examples with both model predictions.
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=pd.read_csv(ROOT/'results/predictions/vision_vit.csv'); b=pd.read_csv(ROOT/'results/predictions/vision_resnet.csv')
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for j,ax in enumerate(axes.flat):
        i=int(a.iloc[j]['id']); img,label=test_base[i]
        ax.imshow(img); ax.axis('off')
        ax.set_title(f'True: {base.classes[label]}\nViT: {base.classes[int(a.iloc[j].predicted)]}\nResNet: {base.classes[int(b.iloc[j].predicted)]}',fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT/'results/figures/vision_examples.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
