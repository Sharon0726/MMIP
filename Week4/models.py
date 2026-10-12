"""Models kept separate so trained weights can also be used for inference."""
import torch
from torch import nn


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, kind='rnn', embedding=128, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        self.recurrent = (nn.RNN if kind == 'rnn' else nn.LSTM)(embedding, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        # Select the last REAL token, never a padded position. This allows cuDNN batching.
        out, _ = self.recurrent(self.dropout(self.embedding(x)))
        last = out[torch.arange(len(x), device=x.device), lengths - 1]
        return self.head(self.dropout(last))


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, kind='lstm', embedding=256, hidden=256):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        self.h0 = nn.Linear(512, hidden)
        self.c0 = nn.Linear(512, hidden) if kind == 'lstm' else None
        self.recurrent = (nn.LSTM if kind == 'lstm' else nn.GRU)(embedding, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, vocab_size)

    def initial(self, features):
        h = torch.tanh(self.h0(features)).unsqueeze(0)
        return (h, torch.tanh(self.c0(features)).unsqueeze(0)) if self.kind == 'lstm' else h

    def forward(self, features, inputs):
        out, _ = self.recurrent(self.dropout(self.embedding(inputs)), self.initial(features))
        return self.head(self.dropout(out))

    @torch.inference_mode()
    def generate(self, features, max_length=32):
        state = self.initial(features)
        current = torch.full((len(features), 1), 2, dtype=torch.long, device=features.device)
        done = torch.zeros(len(features), dtype=torch.bool, device=features.device)
        outputs = []
        for _ in range(max_length - 1):
            out, state = self.recurrent(self.embedding(current), state)
            logits = self.head(out[:, -1])
            logits[:, [0, 1, 2]] = -torch.inf
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, 3), nxt)
            outputs.append(nxt)
            done |= nxt.eq(3)
            current = nxt[:, None]
            if done.all():
                break
        return torch.stack(outputs, dim=1)


def make_vision(kind, pretrained=True):
    import timm
    from torchvision.models import ResNet18_Weights, resnet18
    if kind == 'vit':
        model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained, num_classes=10)
        train_prefixes = ('blocks.11.', 'norm.', 'head.')
    else:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        train_prefixes = ('layer4.', 'fc.')
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(train_prefixes)
    return model


def vision_train_mode(model, kind):
    model.eval()  # Frozen stages (including BN statistics) stay fixed.
    if kind == 'vit':
        model.blocks[-1].train()
        model.norm.train()
        model.head.train()
    else:
        model.layer4.train()
        model.fc.train()
