"""Retrieval metric learning and image/text fusion models."""
import torch
from torch import nn
from torch.nn import functional as F


def backbone(kind, pretrained=True):
    if kind == 'resnet':
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    else:
        import timm
        net = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k',pretrained=pretrained,num_classes=0)
        dim = net.num_features
    return net, dim


class RetrievalModel(nn.Module):
    def __init__(self, kind, dimension=128, pretrained=True):
        super().__init__()
        self.kind = kind
        self.backbone, dim = backbone(kind, pretrained)
        prefixes = ('blocks.11.','norm.') if kind=='vit' else ('layer4.',)
        for name, p in self.backbone.named_parameters():
            p.requires_grad = name.startswith(prefixes)
        self.project = nn.Linear(dim, dimension)
        self.classifier = nn.Linear(dimension, 10)
    def forward(self, images):
        z = F.normalize(self.project(self.backbone(images)),dim=-1)
        return z, self.classifier(z)
    def training_mode(self):
        self.train()
        self.backbone.eval()
        if self.kind=='vit':
            self.backbone.blocks[-1].train(); self.backbone.norm.train()
        else:
            self.backbone.layer4.train()


def supervised_contrastive(z, labels, temperature=.1):
    """One-view supervised contrastive loss; self-pairs excluded from both sums."""
    n=len(z); diagonal=torch.eye(n,dtype=torch.bool,device=z.device)
    positives=labels[:,None].eq(labels[None,:]) & ~diagonal
    logits=(z.float() @ z.float().T)/temperature
    logits=logits.masked_fill(diagonal,-1e9)
    logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    counts=positives.sum(1); valid=counts>0
    if not valid.any(): return z.sum()*0
    return -((logp*positives).sum(1)/counts.clamp_min(1))[valid].mean()


class FusionModel(nn.Module):
    def __init__(self,image_dim,text_dim=384,hidden=256,mode='fusion'):
        super().__init__(); self.mode=mode
        self.image=nn.Sequential(nn.LayerNorm(image_dim),nn.Linear(image_dim,hidden),nn.ReLU())
        self.text=nn.Sequential(nn.LayerNorm(text_dim),nn.Linear(text_dim,hidden),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(hidden*(4 if mode=='fusion' else 1),hidden),nn.ReLU(),nn.Dropout(.3),nn.Linear(hidden,2))
    def forward(self,images,texts):
        u,v=self.image(images),self.text(texts)
        fused=torch.cat([u,v,(u-v).abs(),u*v],dim=-1) if self.mode=='fusion' else (u if self.mode=='image' else v)
        return self.head(fused)
