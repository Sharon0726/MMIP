import torch
from torch import nn

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


