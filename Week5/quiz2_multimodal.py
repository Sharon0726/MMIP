"""Learn binary image/text original-pair classification using two backbone combinations."""
import time
import textwrap
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from common import ROOT,CFG,DEVICE,seed_all,sync,save_json,classification_metrics,tokens
from data_utils import flickr_partitions
from features import image_features,text_features
from models import FusionModel


def make_pairs(ids,texts):
    """Within-split bijection balances every image and text across the two labels."""
    rng=np.random.default_rng(CFG['seed']); rows=[]
    for split,values in ids.items():
        arr=np.array(values)
        for attempt in range(10000):
            shifted=rng.permutation(arr)
            if np.all(shifted!=arr) and all(tokens(texts[int(a)])!=tokens(texts[int(b)]) for a,b in zip(arr,shifted)):
                break
        else: raise RuntimeError('Could not make a non-identical caption permutation.')
        for a,b in zip(arr,shifted):
            rows.extend([{'split':split,'image_id':int(a),'text_id':int(a),'label':1},
                         {'split':split,'image_id':int(a),'text_id':int(b),'label':0}])
    return pd.DataFrame(rows)


def main():
    seed_all(CFG['seed']); c=CFG['fusion']; ds,ids=flickr_partitions()
    texts=[captions[0] for captions in ds['captions']]
    pairs=make_pairs(ids,texts); pairs.to_csv(ROOT/'results/multimodal_pairs.csv',index=False)
    tf=text_features(texts); records=[]
    save_json('results/multimodal_data.json',{'task':'binary original image-caption pairing (1=original, 0=within-split permutation)',
       'caption_selection':'first human caption per image, no reference text used as label string',
       'counts':pairs.groupby('split').size().to_dict(),'labels':{'0':'permuted pair','1':'original pair'},
       'negative_caveat':'A different-image caption can still be semantically plausible. This is noisy correspondence supervision, not human-verified semantic truth.',
       'balance':'Each image and each text occurs exactly once per label within each split.',
       'backbones':'fresh frozen ImageNet ResNet18/ViT-Tiny plus frozen all-MiniLM-L6-v2; not Quiz1 fine-tuned weights'})
    for kind,mode in [('resnet','fusion'),('vit','fusion'),('resnet','image'),('resnet','text')]:
        seed_all(CFG['seed']); vf=image_features(kind)
        loaders={}
        for split in ids:
            p=pairs.loc[pairs.split==split]
            loaders[split]=DataLoader(TensorDataset(vf[p.image_id.to_numpy()],tf[p.text_id.to_numpy()],torch.tensor(p.label.to_numpy())),
                batch_size=c['batch_size'],shuffle=split=='train')
        model=FusionModel(vf.shape[1],tf.shape[1],c['hidden'],mode).to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.01)
        best=-1; history=[]; tag=f'{kind}_{mode}'; epochs=c['epochs'] if mode=='fusion' else 10
        sync(); start=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train(); total=0; n=0
            for image,text,y in loaders['train']:
                image,text,y=image.to(DEVICE),text.to(DEVICE),y.to(DEVICE); opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(image,text),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step(); total+=loss.item()*len(y); n+=len(y)
            model.eval(); probs=[]; labels=[]
            with torch.inference_mode():
                for image,text,y in loaders['validation']:
                    probs.append(model(image.to(DEVICE),text.to(DEVICE)).softmax(-1).cpu().numpy()); labels.extend(y.tolist())
            val=classification_metrics(np.array(labels),np.concatenate(probs))
            history.append({'epoch':epoch,'train_loss':total/n,'val_macro_f1':val['macro_f1'],'val_auc':val['auc']})
            if val['macro_f1']>best:
                best=val['macro_f1']; best_epoch=epoch; torch.save(model.state_dict(),ROOT/f'checkpoints/multimodal_{tag}.pt')
            if epoch==1 or epoch%5==0: print(f'multimodal {tag} epoch={epoch} val_f1={val["macro_f1"]:.4f}',flush=True)
        sync(); seconds=time.perf_counter()-start
        pd.DataFrame(history).to_csv(ROOT/f'results/history/multimodal_{tag}.csv',index=False)
        model.load_state_dict(torch.load(ROOT/f'checkpoints/multimodal_{tag}.pt',weights_only=True,map_location=DEVICE)); model.eval()
        probs=[]
        with torch.inference_mode():
            for image,text,y in loaders['test']:
                probs.append(model(image.to(DEVICE),text.to(DEVICE)).softmax(-1).cpu().numpy())
        probs=np.concatenate(probs); pred=pairs.loc[pairs.split=='test'].copy().reset_index(drop=True)
        pred['prediction']=probs.argmax(1); pred['p_0']=probs[:,0]; pred['p_1']=probs[:,1]
        pred.to_csv(ROOT/f'results/predictions/multimodal_{tag}.csv',index=False)
        row={'model':tag,'best_epoch':best_epoch,'train_seconds':seconds,'trainable_parameters':sum(p.numel() for p in model.parameters()),
             'image_backbone':kind,'text_backbone':'all-MiniLM-L6-v2','mode':mode,**classification_metrics(pred.label.to_numpy(),probs)}
        records.append(row); print(row,flush=True)
        if mode=='fusion':
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(3,2,figsize=(12,13)); examples=[]
            for j,ax in enumerate(axes.flat):
                r=pred.iloc[j]; image_id,text_id=int(r.image_id),int(r.text_id)
                ax.imshow(ds[image_id]['image']); ax.axis('off')
                ax.set_title(textwrap.fill(texts[text_id],55)+f'\nTrue={int(r.label)} Pred={int(r.prediction)} P(original)={r.p_1:.3f}',fontsize=10)
                examples.append({'image_id':image_id,'text_id':text_id,'text':texts[text_id],'label':int(r.label),'prediction':int(r.prediction),'p_1':float(r.p_1)})
            fig.tight_layout(); fig.savefig(ROOT/f'results/figures/multimodal_{kind}.png',dpi=130); plt.close(fig)
            save_json(f'results/multimodal_examples_{kind}.json',examples)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(ROOT/'results/multimodal_metrics.csv',index=False)


if __name__=='__main__': main()
