"""Run a downloaded Week5 checkpoint on a new image (CPU or GPU)."""
import argparse
import json
import torch
from torch.nn import functional as F
from PIL import Image
from common import ROOT,CFG,DEVICE,seed_all
from data_utils import image_transform
from models import RetrievalModel,FusionModel,backbone
from caption_decoder import CaptionModel
from quiz3_clip_caption import decode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['retrieval','multimodal','caption']); p.add_argument('image')
    p.add_argument('--backbone',choices=['resnet','vit'],default='vit')
    p.add_argument('--text',default=''); p.add_argument('--decoder',choices=['gru','lstm'],default='gru')
    args=p.parse_args(); seed_all(CFG['seed']); img=Image.open(args.image).convert('RGB')
    with torch.inference_mode():
        if args.task=='retrieval':
            model=RetrievalModel(args.backbone,CFG['retrieval']['embedding'],pretrained=False).to(DEVICE).eval()
            model.load_state_dict(torch.load(ROOT/f'checkpoints/retrieval_{args.backbone}.pt',map_location=DEVICE,weights_only=True))
            gallery=torch.load(ROOT/f'checkpoints/gallery_{args.backbone}.pt',map_location=DEVICE,weights_only=True)
            z,_=model(image_transform(args.backbone)(img).unsqueeze(0).to(DEVICE))
            scores,indices=(z @ gallery['embeddings'].T).topk(5,dim=1)
            print(json.dumps([{'official_train_id':int(gallery['source_ids'][i]),'label':int(gallery['labels'][i]),'cosine':float(s)}
                  for s,i in zip(scores[0],indices[0])],indent=2))
        elif args.task=='multimodal':
            if not args.text: p.error('--text is required for multimodal classification')
            from transformers import AutoTokenizer,AutoModel
            name='sentence-transformers/all-MiniLM-L6-v2'; revision=CFG['revisions'][name]
            tokenizer=AutoTokenizer.from_pretrained(name,revision=revision)
            text_model=AutoModel.from_pretrained(name,revision=revision).to(DEVICE).eval()
            batch=tokenizer([args.text],padding=True,truncation=True,max_length=CFG['fusion']['text_max_length'],return_tensors='pt').to(DEVICE)
            h=text_model(**batch).last_hidden_state; mask=batch.attention_mask[:,:,None]
            text=F.normalize((h*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
            image_model,dim=backbone(args.backbone); image_model=image_model.to(DEVICE).eval()
            image=image_model(image_transform(args.backbone)(img).unsqueeze(0).to(DEVICE))
            model=FusionModel(dim,384,CFG['fusion']['hidden']).to(DEVICE).eval()
            model.load_state_dict(torch.load(ROOT/f'checkpoints/multimodal_{args.backbone}_fusion.pt',map_location=DEVICE,weights_only=True))
            probs=model(image,text).softmax(-1)[0].tolist()
            print(json.dumps({'p_permuted':probs[0],'p_original':probs[1],'prediction':int(probs[1]>=probs[0]),
                             'note':'Model trained on original vs permuted Flickr8k pairs; not a calibrated semantic truth detector.'},indent=2))
        else:
            from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
            name='openai/clip-vit-base-patch32'; revision=CFG['revisions'][name]
            encoder=CLIPVisionModelWithProjection.from_pretrained(name,revision=revision).to(DEVICE).eval()
            processor=CLIPImageProcessor.from_pretrained(name,revision=revision)
            f=encoder(**processor(images=[img],return_tensors='pt').to(DEVICE)).image_embeds
            vocab=json.loads((ROOT/'baseline_week4/caption_vocab.json').read_text()); c=CFG['caption']
            decoder=CaptionModel(len(vocab),args.decoder,c['embedding'],c['hidden']).to(DEVICE).eval()
            decoder.load_state_dict(torch.load(ROOT/f'checkpoints/caption_clip_{args.decoder}.pt',map_location=DEVICE,weights_only=True))
            print(decode(decoder.generate(f,c['max_length'])[0].tolist(),{v:k for k,v in vocab.items()}))


if __name__=='__main__': main()
