"""Train CLIP-conditioned decoders and compare exact Week4 test predictions with BLEU/BERTScore."""
import json
import time
import textwrap
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from nltk.translate.bleu_score import corpus_bleu
from common import ROOT,CFG,DEVICE,seed_all,sync,save_json,encode,tokens
from data_utils import flickr_partitions
from features import image_features
from caption_decoder import CaptionModel


class CaptionRows(Dataset):
    def __init__(self,features,ids,captions,vocab,length):
        self.features=features
        self.rows=[(i,torch.tensor(encode(t,vocab,length,caption=True)[0])) for i in ids for t in captions[i]]
    def __len__(self): return len(self.rows)
    def __getitem__(self,j):
        i,t=self.rows[j]; return self.features[i],t


def decode(ids,inverse):
    words=[]
    for i in ids:
        if i==3: break
        if i>3: words.append(inverse[i])
    return ' '.join(words)


def bleu(rows):
    refs=[[tokens(t) for t in r['references']] for r in rows]
    hyp=[tokens(r['generated']) for r in rows]
    return {'bleu1':corpus_bleu(refs,hyp,weights=(1,0,0,0)),
            'bleu4':corpus_bleu(refs,hyp,weights=(.25,)*4)}


def main():
    seed_all(CFG['seed']); c=CFG['caption']; ds,ids=flickr_partitions()
    captions=ds['captions']; names=ds['image_filename']; features=image_features('clip')
    vocab=json.loads((ROOT/'baseline_week4/caption_vocab.json').read_text()); inverse={v:k for k,v in vocab.items()}
    save_json('results/caption_vocab.json',vocab)
    save_json('results/caption_data.json',{'encoder':'CLIP ViT-B/32 frozen image tower + visual projection (512 dimensions)',
        'encoder_revision':CFG['revisions']['openai/clip-vit-base-patch32'],'vocabulary':'identical Week4 training-only vocabulary',
        'vocab_size':len(vocab),'max_length':c['max_length'],'counts':{s:len(v) for s,v in ids.items()},
        'comparison':'same split, references, tokenizer, vocabulary, GRU/LSTM architecture, epochs and greedy decoding; image encoder changed'})
    predictions={}; metrics=[]
    for kind in ['gru','lstm']:
        seed_all(CFG['seed']); tag=f'clip_{kind}'
        loaders={s:DataLoader(CaptionRows(features,ids[s],captions,vocab,c['max_length']),batch_size=c['batch_size'],shuffle=s=='train') for s in ['train','validation']}
        model=CaptionModel(len(vocab),kind,c['embedding'],c['hidden']).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=c['lr']); criterion=nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
        best=float('inf'); history=[]; sync(); start=time.perf_counter()
        for epoch in range(1,c['epochs']+1):
            losses={}
            for split,loader in loaders.items():
                model.train(split=='train'); total=0; count=0
                for f,cap in loader:
                    f,cap=f.to(DEVICE),cap.to(DEVICE); target=cap[:,1:]; n=target.ne(0).sum()
                    with torch.set_grad_enabled(split=='train'):
                        logits=model(f,cap[:,:-1]); loss=criterion(logits.reshape(-1,len(vocab)),target.reshape(-1))
                        if split=='train':
                            opt.zero_grad(set_to_none=True); (loss/n).backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                    total+=loss.item(); count+=n.item()
                losses[split]=total/count
            history.append({'epoch':epoch,'train_loss':losses['train'],'val_loss':losses['validation']})
            print(f'{tag} epoch={epoch} val_loss={losses["validation"]:.4f}',flush=True)
            if losses['validation']<best:
                best=losses['validation']; best_epoch=epoch
                torch.save(model.state_dict(),ROOT/f'checkpoints/caption_{tag}.pt')
        sync(); elapsed=time.perf_counter()-start
        pd.DataFrame(history).to_csv(ROOT/f'results/history/caption_{tag}.csv',index=False)
        model.load_state_dict(torch.load(ROOT/f'checkpoints/caption_{tag}.pt',map_location=DEVICE,weights_only=True)); model.eval()
        generated=[]
        with torch.inference_mode():
            for (f,) in DataLoader(TensorDataset(features[ids['test']]),batch_size=64):
                generated.extend(decode(g,inverse) for g in model.generate(f.to(DEVICE),c['max_length']).cpu().tolist())
        rows=[{'source_id':int(i),'filename':names[i],'generated':g,'references':captions[i]} for i,g in zip(ids['test'],generated)]
        save_json(f'results/predictions/caption_{tag}.json',rows); predictions[tag]=rows
        metrics.append({'model':tag,'week':5,'best_epoch':best_epoch,'train_seconds':elapsed,
                        'decoder_parameters':sum(p.numel() for p in model.parameters()),**bleu(rows)})
        del model; torch.cuda.empty_cache()
    for kind in ['gru','lstm']:
        tag=f'resnet_{kind}'; rows=json.loads((ROOT/f'baseline_week4/caption_{kind}.json').read_text())
        for old,new in zip(rows,predictions['clip_gru']):
            assert old['filename']==new['filename'] and old['source_id']==new['source_id'] and old['references']==new['references']
        assert len(rows)==len(ids['test'])
        predictions[tag]=rows
        metrics.append({'model':tag,'week':4,**bleu(rows)})
    # Use the same pinned contextual model and settings for every system.
    from huggingface_hub import snapshot_download
    from bert_score import BERTScorer
    from bert_score.utils import get_hash
    bert_name='google-bert/bert-base-uncased'
    bert_path=snapshot_download(bert_name,revision=CFG['revisions'][bert_name],
        allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json'])
    scorer=BERTScorer(model_type=bert_path,num_layers=9,device=str(DEVICE),batch_size=32,
        idf=False,rescale_with_baseline=False,use_fast_tokenizer=False,nthreads=2)
    bert_outputs={}
    for row in metrics:
        tag=row['model']; data=predictions[tag]
        p,r,f=scorer.score([x['generated'] for x in data],[x['references'] for x in data],batch_size=32)
        frame=pd.DataFrame({'source_id':[x['source_id'] for x in data],'precision':p.cpu().numpy(),'recall':r.cpu().numpy(),'f1':f.cpu().numpy()})
        frame.to_csv(ROOT/f'results/predictions/bertscore_{tag}.csv',index=False); bert_outputs[tag]=frame.f1.to_numpy()
        row.update(bertscore_precision=float(p.mean()),bertscore_recall=float(r.mean()),bertscore_f1=float(f.mean()),test_images=len(data))
        print(row,flush=True)
    save_json('results/bertscore_settings.json',{'model':bert_name,'revision':CFG['revisions'][bert_name],
        'num_layers':9,'idf':False,'rescale_with_baseline':False,'use_fast_tokenizer':False,'batch_size':32,
        'multiple_references':'BERTScore 0.3.13 max across all five references for each P/R/F component, then mean over images',
        'hash':get_hash(bert_name,9,False,False,False,False)})
    # Paired bootstrap describes uncertainty across the fixed test images, not training seeds.
    rng=np.random.default_rng(CFG['seed']); intervals=[]
    for kind in ['gru','lstm']:
        diff=bert_outputs[f'clip_{kind}']-bert_outputs[f'resnet_{kind}']
        means=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(1)
        intervals.append({'decoder':kind,'mean_delta_bertscore_f1':float(diff.mean()),
            'paired_bootstrap_ci95_low':float(np.percentile(means,2.5)),'paired_bootstrap_ci95_high':float(np.percentile(means,97.5))})
    pd.DataFrame(intervals).to_csv(ROOT/'results/caption_paired_comparison.csv',index=False)
    pd.DataFrame(metrics).to_csv(ROOT/'results/caption_comparison.csv',index=False)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(14,15))
    examples=[]
    for j,ax in enumerate(axes.flat):
        i=ids['test'][j]; img=ds[i]['image'].convert('RGB'); img.thumbnail((640,640))
        img.save(ROOT/f'results/figures/caption_sample_{j+1}.jpg'); ax.imshow(img); ax.axis('off')
        lines=[f'#{j+1} | {names[i]}']
        for tag in ['resnet_gru','resnet_lstm','clip_gru','clip_lstm']:
            lines.append(f'{tag}: '+textwrap.fill(predictions[tag][j]['generated'],58))
        ax.set_title('\n'.join(lines),fontsize=9)
        examples.append({'source_id':i,'filename':names[i],**{tag:predictions[tag][j]['generated'] for tag in predictions},'references':captions[i]})
    fig.tight_layout(); fig.savefig(ROOT/'results/figures/caption_examples.png',dpi=140); plt.close(fig)
    save_json('results/caption_examples.json',examples)


if __name__=='__main__': main()
