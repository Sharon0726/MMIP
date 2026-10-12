"""Extract frozen image and language backbone features, with cache identity checks."""
import hashlib
import json
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import CFG, ROOT, DEVICE, sync, save_json
from data_utils import flickr, ImageRows, image_transform
from models import backbone


def image_features(kind):
    data=flickr(); name='openai/clip-vit-base-patch32'
    identity={'kind':kind,'dataset_revision':CFG['revisions']['tsystems/flickr8k'],
              'model_revision':CFG['revisions'][name] if kind=='clip' else 'ImageNet pretrained; timm 1.0.15 / torchvision V1',
              'transform':'CLIPImageProcessor original' if kind=='clip' else 'resize224 + ImageNet/model normalization'}
    path=ROOT/f'cache/flickr_{kind}.pt'
    if path.exists():
        saved=torch.load(path,weights_only=True)
        assert saved['identity']==identity
        return saved['features']
    if kind=='clip':
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        model=CLIPVisionModelWithProjection.from_pretrained(name,revision=CFG['revisions'][name]).to(DEVICE).eval()
        processor=CLIPImageProcessor.from_pretrained(name,revision=CFG['revisions'][name])
        # Batch the PIL processing in the parent process; no lambda in multiprocessing.
        batches=(list(range(i,min(i+64,len(data)))) for i in range(0,len(data),64))
    else:
        model,_=backbone(kind); model=model.to(DEVICE).eval()
        batches=DataLoader(ImageRows(data,range(len(data)),image_transform(kind),'image',None),batch_size=64,num_workers=2)
    for p in model.parameters(): p.requires_grad=False
    out=[]; sync(); start=time.perf_counter()
    with torch.inference_mode():
        for step,batch in enumerate(batches):
            if kind=='clip':
                x=processor(images=[data[i]['image'].convert('RGB') for i in batch],return_tensors='pt')['pixel_values'].to(DEVICE)
                f=model(pixel_values=x).image_embeds
            else:
                f=model(batch[0].to(DEVICE))
            out.append(f.cpu())
            if step%30==0: print(f'{kind} frozen features: {step*64}/{len(data)}',flush=True)
    sync(); elapsed=time.perf_counter()-start
    features=torch.cat(out)
    torch.save({'identity':identity,'features':features},path)
    save_json(f'results/features_{kind}.json',{**identity,'shape':list(features.shape),'extraction_seconds':elapsed,'frozen':True})
    del model; torch.cuda.empty_cache()
    return features


def text_features(texts):
    name='sentence-transformers/all-MiniLM-L6-v2'
    identity={'model':name,'revision':CFG['revisions'][name],
              'max_length':CFG['fusion']['text_max_length'],
              'text_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()}
    path=ROOT/'cache/minilm_features.pt'
    if path.exists():
        saved=torch.load(path,weights_only=True); assert saved['identity']==identity
        return saved['features']
    from transformers import AutoTokenizer,AutoModel
    tokenizer=AutoTokenizer.from_pretrained(name,revision=identity['revision'])
    model=AutoModel.from_pretrained(name,revision=identity['revision']).to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad=False
    out=[]; sync(); start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(texts),64):
            batch=tokenizer(texts[i:i+64],padding=True,truncation=True,max_length=identity['max_length'],return_tensors='pt').to(DEVICE)
            hidden=model(**batch).last_hidden_state; mask=batch.attention_mask[:,:,None]
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            out.append(F.normalize(pooled,dim=-1).cpu())
    sync(); elapsed=time.perf_counter()-start
    features=torch.cat(out); torch.save({'identity':identity,'features':features},path)
    save_json('results/features_minilm.json',{**identity,'shape':list(features.shape),'extraction_seconds':elapsed,'frozen':True,'pooling':'attention-mask mean; L2 normalize'})
    del model; torch.cuda.empty_cache()
    return features
