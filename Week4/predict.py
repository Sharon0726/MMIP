"""Use trained checkpoints on a new sentence or image (CPU supported)."""
import argparse
import json
import torch
from PIL import Image
from common import ROOT, CFG, DEVICE, encode
from models import SentimentModel, CaptionModel, make_vision


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['sentiment','vision','caption'])
    p.add_argument('input',help='Text for sentiment; image path for vision/caption')
    p.add_argument('--model',default=None)
    args=p.parse_args()
    kind=args.model or {'sentiment':'lstm','vision':'vit','caption':'lstm'}[args.task]
    c=CFG[args.task]
    if args.task=='sentiment':
        assert kind in ['rnn','lstm']
        vocab=json.loads((ROOT/'results/sentiment_vocab.json').read_text())
        model=SentimentModel(len(vocab),kind,c['embedding'],c['hidden']).to(DEVICE)
        model.load_state_dict(torch.load(ROOT/f'checkpoints/sentiment_{kind}.pt',map_location=DEVICE,weights_only=True)); model.eval()
        ids,n=encode(args.input,vocab,c['max_length'])
        with torch.inference_mode():
            probs=model(torch.tensor([ids],device=DEVICE),torch.tensor([n],device=DEVICE)).softmax(-1)[0].tolist()
        print(json.dumps({'negative':probs[0],'positive':probs[1],'prediction':'positive' if probs[1]>=probs[0] else 'negative'},indent=2))
    elif args.task=='vision':
        assert kind in ['vit','resnet']
        from quiz3_vision import image_transform
        model=make_vision(kind,pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ROOT/f'checkpoints/vision_{kind}.pt',map_location=DEVICE,weights_only=True)); model.eval()
        image=image_transform(kind)(Image.open(args.input).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.inference_mode(): probs=model(image).softmax(-1)[0].tolist()
        classes=json.loads((ROOT/'results/vision_data.json').read_text())['classes']
        print(json.dumps({'prediction':classes[max(range(10),key=lambda i:probs[i])],'probabilities':dict(zip(classes,probs))},indent=2))
    else:
        assert kind in ['gru','lstm']
        from torchvision.models import resnet18,ResNet18_Weights
        from quiz4_caption import decode
        vocab=json.loads((ROOT/'results/caption_vocab.json').read_text()); inverse={v:k for k,v in vocab.items()}
        weights=ResNet18_Weights.IMAGENET1K_V1
        encoder=resnet18(weights=weights); encoder.fc=torch.nn.Identity(); encoder=encoder.to(DEVICE).eval()
        model=CaptionModel(len(vocab),kind,c['embedding'],c['hidden']).to(DEVICE)
        model.load_state_dict(torch.load(ROOT/f'checkpoints/caption_{kind}.pt',map_location=DEVICE,weights_only=True)); model.eval()
        image=weights.transforms()(Image.open(args.input).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.inference_mode(): ids=model.generate(encoder(image),c['max_length'])[0].tolist()
        print(decode(ids,inverse))


if __name__=='__main__': main()
