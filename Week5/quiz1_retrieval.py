"""Train CNN/ViT embeddings with supervised contrastive loss; evaluate held-out retrieval."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from common import ROOT, CFG, DEVICE, seed_all, sync, save_json
from data_utils import cifar, ImageRows, image_transform
from models import RetrievalModel, supervised_contrastive


@torch.inference_mode()
def embed(model, loader):
    model.eval(); features=[]; labels=[]
    for x,y in loader:
        # Full precision normalized embeddings are used for ranking.
        z,_=model(x.to(DEVICE)); features.append(z.cpu()); labels.extend(y.tolist())
    return torch.cat(features),np.array(labels)


def retrieval_metrics(query_labels, neighbor_labels):
    relevant=neighbor_labels==query_labels[:,None]
    precision=np.cumsum(relevant,axis=1)/np.arange(1,relevant.shape[1]+1)
    return {'hit_at_1':float(relevant[:,:1].any(1).mean()),'hit_at_5':float(relevant[:,:5].any(1).mean()),
            'hit_at_10':float(relevant.any(1).mean()),'precision_at_5':float(relevant[:,:5].mean()),
            'map_at_10':float(((precision*relevant).sum(1)/10).mean())}


@torch.inference_mode()
def rank_queries(q,g,exclude_self=False):
    gallery=g.to(DEVICE); scores=[]; indices=[]
    for start in range(0,len(q),256):
        sim=q[start:start+256].to(DEVICE) @ gallery.T
        if exclude_self:
            sim[torch.arange(len(sim),device=DEVICE),torch.arange(start,start+len(sim),device=DEVICE)]=-torch.inf
        score,index=sim.topk(10,dim=1)
        scores.append(score.cpu().numpy()); indices.append(index.cpu().numpy())
    return np.concatenate(indices),np.concatenate(scores)


def main():
    seed_all(CFG['seed']); c=CFG['retrieval']; ds=cifar()
    labels=np.array(ds['train']['label']); rng=np.random.default_rng(CFG['seed'])
    ids={s:[] for s in ['train','validation','gallery']}
    for cls in range(10):
        order=rng.permutation(np.flatnonzero(labels==cls))
        a,b,d=c['train_per_class'],c['val_per_class'],c['gallery_per_class']
        ids['train'].extend(order[:a]); ids['validation'].extend(order[a:a+b]); ids['gallery'].extend(order[a+b:a+b+d])
    ids={s:sorted(map(int,v)) for s,v in ids.items()}; ids['query']=list(range(len(ds['test'])))
    assert all(not set(ids[a]) & set(ids[b]) for a,b in [('train','validation'),('train','gallery'),('validation','gallery')])
    rows=[]
    for s,v in ids.items():
        src='test' if s=='query' else 'train'
        rows.extend({'split':s,'source':src,'source_id':i,'label':int(ds[src][i]['label'])} for i in v)
    pd.DataFrame(rows).to_csv(ROOT/'results/retrieval_split.csv',index=False)
    classes=ds['train'].features['label'].names
    save_json('results/retrieval_data.json',{'dataset':'uoft-cs/cifar10','revision':CFG['revisions']['uoft-cs/cifar10'],
       'classes':classes,'counts':{s:len(v) for s,v in ids.items()},'relevance':'same CIFAR-10 class; semantic category retrieval, not instance identity',
       'gallery_relevant_per_class':c['gallery_per_class'],'unused_official_train':len(ds['train'])-sum(len(ids[s]) for s in ['train','validation','gallery'])})
    all_metrics=[]
    for kind in ['resnet','vit']:
        seed_all(CFG['seed'])
        loaders={s:DataLoader(ImageRows(ds['test' if s=='query' else 'train'],v,image_transform(kind,s=='train')),
             batch_size=c['batch_size'],shuffle=s=='train',num_workers=2,pin_memory=DEVICE.type=='cuda') for s,v in ids.items()}
        model=RetrievalModel(kind,c['embedding']).to(DEVICE)
        probe='backbone.blocks.11.attn.qkv.weight' if kind=='vit' else 'backbone.layer4.0.conv1.weight'
        before=dict(model.named_parameters())[probe].detach().cpu().clone()
        opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=c['lr'],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda'); history=[]; best=-1
        sync(); start=time.perf_counter()
        for epoch in range(1,c['epochs']+1):
            model.training_mode(); total=0; n=0
            for x,y in loaders['train']:
                x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type,enabled=DEVICE.type=='cuda'):
                    z,logits=model(x); ce=nn.functional.cross_entropy(logits,y)
                loss=supervised_contrastive(z,y,c['temperature'])+.5*ce
                scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt); scaler.update(); total+=loss.item()*len(y); n+=len(y)
            val,vy=embed(model,loaders['validation']); neighbors,_=rank_queries(val,val,exclude_self=True)
            hit=float((vy[neighbors[:,0]]==vy).mean())
            history.append({'epoch':epoch,'train_loss':total/n,'validation_hit_at_1':hit})
            print(f'retrieval {kind} epoch={epoch} val_hit@1={hit:.4f}',flush=True)
            if hit>best:
                best=hit; best_epoch=epoch
                torch.save(model.state_dict(),ROOT/f'checkpoints/retrieval_{kind}.pt')
        sync(); train_seconds=time.perf_counter()-start
        pd.DataFrame(history).to_csv(ROOT/f'results/history/retrieval_{kind}.csv',index=False)
        model.load_state_dict(torch.load(ROOT/f'checkpoints/retrieval_{kind}.pt',map_location=DEVICE,weights_only=True))
        delta=(dict(model.named_parameters())[probe].detach().cpu()-before).abs().max().item()
        assert delta>0
        gallery,gy=embed(model,loaders['gallery']); queries,qy=embed(model,loaders['query'])
        index,score=rank_queries(queries,gallery)
        gallery_ids=np.array(ids['gallery']); neighbors=gy[index]
        np.savez_compressed(ROOT/f'results/predictions/retrieval_{kind}.npz',query_ids=np.array(ids['query']),query_labels=qy,
            gallery_ids=gallery_ids[index],neighbor_labels=neighbors,similarities=score)
        torch.save({'embeddings':gallery,'source_ids':torch.tensor(ids['gallery']),'labels':torch.tensor(gy)},ROOT/f'checkpoints/gallery_{kind}.pt')
        times=[]; model.eval(); gd=gallery.to(DEVICE); transform=image_transform(kind)
        with torch.inference_mode():
            for i in ids['query'][:103]:
                img=ds['test'][i]['img'].convert('RGB'); sync(); t=time.perf_counter()
                z,_=model(transform(img).unsqueeze(0).to(DEVICE)); (z @ gd.T).topk(10,dim=1); sync()
                if i>=3: times.append((time.perf_counter()-t)*1000)
        pd.DataFrame({'query_id':ids['query'][3:103],'milliseconds':times}).to_csv(ROOT/f'results/retrieval_latency_{kind}.csv',index=False)
        row={'model':kind,'best_epoch':best_epoch,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'train_seconds':train_seconds,'mean_query_ms':float(np.mean(times)),'backbone_max_abs_change':delta,**retrieval_metrics(qy,neighbors)}
        all_metrics.append(row); print(row,flush=True)
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(6,6,figsize=(12,12))
        for j in range(6):
            axes[j,0].imshow(ds['test'][j]['img']); axes[j,0].set_title(f'Query {j}: {classes[qy[j]]}',fontsize=9)
            for k in range(5):
                gid=int(gallery_ids[index[j,k]]); ax=axes[j,k+1]; ax.imshow(ds['train'][gid]['img'])
                ax.set_title(f'#{k+1} {classes[neighbors[j,k]]}\ncos={score[j,k]:.3f}',fontsize=9,
                             color='green' if neighbors[j,k]==qy[j] else 'red')
        for ax in axes.flat: ax.axis('off')
        fig.tight_layout(); fig.savefig(ROOT/f'results/figures/retrieval_{kind}.png',dpi=150); plt.close(fig)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(all_metrics).to_csv(ROOT/'results/retrieval_metrics.csv',index=False)


if __name__=='__main__': main()
