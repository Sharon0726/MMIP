"""Small regression checks for padding and autoregressive generation boundaries."""
import unittest
import torch
from models import SentimentModel,CaptionModel
from common import encode,make_vocab


class SequenceInvariants(unittest.TestCase):
    def test_padding_cannot_change_sentiment(self):
        for kind in ['rnn','lstm']:
            model=SentimentModel(10,kind).eval()
            with torch.inference_mode():
                a=model(torch.tensor([[4,5,6]]),torch.tensor([3]))
                b=model(torch.tensor([[4,5,6,0,0,0]]),torch.tensor([3]))
            self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_caption_stops_at_eos(self):
        for kind in ['gru','lstm']:
            model=CaptionModel(10,kind).eval()
            with torch.no_grad():
                model.head.weight.zero_(); model.head.bias.zero_(); model.head.bias[3]=100
            output=model.generate(torch.randn(2,512),32)
            self.assertEqual(output.tolist(),[[3],[3]])

    def test_caption_never_emits_pad_unknown_or_bos(self):
        model=CaptionModel(10).eval()
        with torch.no_grad():
            model.head.weight.zero_(); model.head.bias.zero_(); model.head.bias[:3]=100
            model.head.bias[4]=10
        output=model.generate(torch.randn(1,512),5)
        self.assertEqual(output.tolist(),[[4,4,4,4]])

    def test_train_dictionary_handles_unseen_words_and_empty_text(self):
        vocab=make_vocab(['good film'],20)
        self.assertEqual(encode('unseen',vocab,4),([1,0,0,0],1))
        self.assertEqual(encode('',vocab,4),([1,0,0,0],1))
        ids,n=encode('good film extra',vocab,4,caption=True)
        self.assertEqual(ids,[2,vocab['good'],vocab['film'],3])
        self.assertEqual(n,4)


if __name__=='__main__': unittest.main()
