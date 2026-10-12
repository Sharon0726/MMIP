"""Train image-conditioned GRU/LSTM decoders with cached ResNet18 features."""
import time
import textwrap
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
from common import CFG, DEVICE, ROOT, encode, make_vocab, save_json, seed_all, sync, tokens
from models import CaptionModel


class ImageRows(Dataset):
    def __init__(self,data,transform):
        self.data,self.transform=data,transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.transform(self.data[i]['image'].convert('RGB'))


class CaptionRows(Dataset):
    def __init__(self,features,rows,vocab,length):
        self.features=features
        self.items=[(i,torch.tensor(encode(caption,vocab,length,caption=True)[0])) for i,captions in rows for caption in captions]
    def __len__(self):
        return len(self.items)
    def __getitem__(self,j):
        i,caption=self.items[j]
        return self.features[i],caption


def decode(ids,inverse):
    out=[]
    for i in ids:
        if i==3: break
        if i>3: out.append(inverse[i])
    return ' '.join(out)


def main():
    c=CFG['caption']; seed_all(CFG['seed'])
    ds=load_dataset('tsystems/flickr8k',split='train')
    filenames=ds['image_filename']; captions=ds['captions']
    assert len(set(filenames))==len(filenames), 'Duplicate image filenames require grouping before splitting.'
    order=np.random.default_rng(CFG['seed']).permutation(len(ds))
    n,m=c['train_images'],c['val_images']
    ids={'train':order[:n].tolist(),'validation':order[n:n+m].tolist(),'test':order[n+m:].tolist()}
    pd.DataFrame([{'split':s,'source_id':i,'filename':filenames[i], 'captions':len(captions[i])} for s,v in ids.items() for i in v]).to_csv(ROOT/'results/caption_split.csv',index=False)
    # Image-level partition BEFORE expanding the five captions prevents leakage.
    vocab=make_vocab([t for i in ids['train'] for t in captions[i]],c['vocab_size']); inverse={v:k for k,v in vocab.items()}
    save_json('results/caption_vocab.json',vocab)
    save_json('results/caption_data.json', {'source':'tsystems/flickr8k','fingerprint':ds._fingerprint,
        'counts':{s:len(v) for s,v in ids.items()},'caption_counts':{s:sum(len(captions[i]) for i in v) for s,v in ids.items()},
        'vocab_size':len(vocab),'split':'custom seed-42 image-level 6000/1000/remainder; NOT official Karpathy split',
        'encoder':'ResNet18 ImageNet1K_V1, frozen', 'max_length_including_bos_eos':c['max_length']})
    weights=ResNet18_Weights.IMAGENET1K_V1; transform=weights.transforms()
    encoder=resnet18(weights=weights); encoder.fc=nn.Identity(); encoder=encoder.to(DEVICE).eval()
    feature_path=ROOT/'cache/flickr_features.pt'
    if feature_path.exists():
        features=torch.load(feature_path,weights_only=True)
        feature_seconds=None
    else:
        parts=[]; sync(); start=time.perf_counter()
        with torch.inference_mode():
            for k,x in enumerate(DataLoader(ImageRows(ds,transform),batch_size=64,num_workers=2,pin_memory=True)):
                parts.append(encoder(x.to(DEVICE)).cpu())
                if k%30==0: print(f'Caption features: {k*64}/{len(ds)}',flush=True)
        sync(); feature_seconds=time.perf_counter()-start
        features=torch.cat(parts); torch.save(features,feature_path)
    assert features.shape==(len(ds),512)
    loaders={s:DataLoader(CaptionRows(features,[(i,captions[i]) for i in ids[s]],vocab,c['max_length']),batch_size=c['batch_size'],shuffle=s=='train') for s in ['train','validation']}
    criterion=nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
    metrics=[]; all_predictions={}
    for kind in ['gru','lstm']:
        seed_all(CFG['seed']); model=CaptionModel(len(vocab),kind,c['embedding'],c['hidden']).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=c['lr']); best=float('inf'); history=[]
        sync(); start=time.perf_counter()
        for epoch in range(1,c['epochs']+1):
            losses={}
            for split in ['train','validation']:
                model.train(split=='train'); total,tok_count=0.,0
                for f,cap in loaders[split]:
                    f,cap=f.to(DEVICE),cap.to(DEVICE); target=cap[:,1:]; count=target.ne(0).sum()
                    with torch.set_grad_enabled(split=='train'):
                        logits=model(f,cap[:,:-1]); loss=criterion(logits.reshape(-1,len(vocab)),target.reshape(-1))
                        if split=='train':
                            opt.zero_grad(set_to_none=True); (loss/count).backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                    total+=loss.item(); tok_count+=count.item()
                losses[split]=total/tok_count
            history.append({'epoch':epoch,'train_loss':losses['train'],'val_loss':losses['validation']})
            print(f'caption {kind} epoch {epoch}: val_loss={losses["validation"]:.4f}',flush=True)
            if losses['validation']<best:
                best=losses['validation']; best_epoch=epoch
                torch.save(model.state_dict(),ROOT/f'checkpoints/caption_{kind}.pt')
        sync(); elapsed=time.perf_counter()-start
        pd.DataFrame(history).to_csv(ROOT/f'results/history/caption_{kind}.csv',index=False)
        model.load_state_dict(torch.load(ROOT/f'checkpoints/caption_{kind}.pt',map_location=DEVICE,weights_only=True)); model.eval()
        predictions=[]
        for (f,) in DataLoader(TensorDataset(features[ids['test']]),batch_size=64):
            generated=model.generate(f.to(DEVICE),c['max_length']).cpu().tolist()
            predictions.extend(decode(g,inverse) for g in generated)
        refs=[[tokens(t) for t in captions[i]] for i in ids['test']]
        hyps=[tokens(t) for t in predictions]
        # Single-image timings include transform + H2D + encoder + decoder, exclude disk/network I/O.
        sample_ids=ids['test'][:100]; times=[]
        with torch.inference_mode():
            for i in sample_ids[:3]:
                model.generate(encoder(transform(ds[i]['image'].convert('RGB')).unsqueeze(0).to(DEVICE)),c['max_length'])
            sync()
            for i in sample_ids:
                img=ds[i]['image'].convert('RGB')
                sync(); t=time.perf_counter()
                f=encoder(transform(img).unsqueeze(0).to(DEVICE)); model.generate(f,c['max_length']); sync()
                times.append((time.perf_counter()-t)*1000)
        rows=[{'source_id':i,'filename':filenames[i],'generated':pred,'references':captions[i]} for i,pred in zip(ids['test'],predictions)]
        save_json(f'results/predictions/caption_{kind}.json',rows); all_predictions[kind]=predictions
        pd.DataFrame({'source_id':sample_ids,'milliseconds':times}).to_csv(ROOT/f'results/caption_latency_{kind}.csv',index=False)
        row={'model':f'ResNet18 + {kind.upper()}', 'best_epoch':best_epoch,'train_seconds':elapsed,'feature_extraction_seconds':feature_seconds,
             'decoder_parameters':sum(p.numel() for p in model.parameters()),'test_images':len(predictions),
             'bleu1':corpus_bleu(refs,hyps,weights=(1,0,0,0)), 'bleu4':corpus_bleu(refs,hyps,weights=(.25,)*4),
             'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
             'timing_samples':len(times), 'mean_caption_tokens':float(np.mean([len(t) for t in hyps]))}
        metrics.append(row); print(row,flush=True)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(ROOT/'results/caption_metrics.csv',index=False)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed first six test images chosen independently of correctness, shared with Gemini review.
    fig,axes=plt.subplots(3,2,figsize=(13,14))
    samples=[]
    for j,ax in enumerate(axes.flat):
        i=ids['test'][j]; img=ds[i]['image'].convert('RGB')
        img.thumbnail((640,640)); img.save(ROOT/f'results/figures/caption_sample_{j+1}.jpg')
        ax.imshow(img); ax.axis('off')
        title=f'#{j+1} | {filenames[i]}\nGRU: '+textwrap.fill(all_predictions['gru'][j],55)+'\nLSTM: '+textwrap.fill(all_predictions['lstm'][j],55)
        ax.set_title(title,fontsize=10)
        samples.append({'case':j+1,'source_id':i,'filename':filenames[i],'gru':all_predictions['gru'][j],'lstm':all_predictions['lstm'][j],'references':captions[i]})
    fig.tight_layout(); fig.savefig(ROOT/'results/figures/caption_examples.png',dpi=150); plt.close(fig)
    save_json('results/gemini_cases.json',samples)


if __name__=='__main__':
    main()
