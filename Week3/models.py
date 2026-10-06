"""CNN architectures / CNN 模型定義。"""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class PlainCNN(nn.Module):
    """Three sequential convolution blocks, without residual connections."""
    def __init__(self):
        super().__init__()
        blocks = []
        for cin, cout in [(3, 16), (16, 32), (32, 64)]:
            blocks.extend([nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features(x))


def make_resnet():
    """Frozen ImageNet backbone; train only the new 10-class head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc = nn.Linear(512, 10)
    model.eval()  # Also freeze BatchNorm running statistics.
    return model


def parameter_counts(model):
    return {"parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
