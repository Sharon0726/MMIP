"""Small independent checks for contrastive gradients, self-exclusion and balanced pairing."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from models import supervised_contrastive
from quiz1_retrieval import rank_queries,retrieval_metrics
from quiz2_multimodal import make_pairs


class Invariants(unittest.TestCase):
    def test_contrastive_separation_and_finite_gradient(self):
        y=torch.tensor([0,0,1,1]); good=torch.tensor([[1.,0.],[.9,.1],[-1.,0.],[-.9,-.1]],requires_grad=True)
        loss=supervised_contrastive(F.normalize(good,dim=-1),y)
        bad=supervised_contrastive(F.normalize(good.detach()[[0,2,1,3]],dim=-1),y)
        self.assertLess(loss.item(),bad.item()); loss.backward(); self.assertTrue(torch.isfinite(good.grad).all())
    def test_unique_labels_are_finite(self):
        x=torch.randn(4,3,requires_grad=True); loss=supervised_contrastive(F.normalize(x,dim=-1),torch.arange(4))
        loss.backward(); self.assertTrue(torch.isfinite(x.grad).all())
    def test_ranking_excludes_self(self):
        x=F.normalize(torch.randn(12,8),dim=-1); ids,scores=rank_queries(x,x,exclude_self=True)
        self.assertFalse((ids==np.arange(12)[:,None]).any())
    def test_ap_penalizes_late_relevant(self):
        y=np.array([1]); ranks=np.array([[0,1,1,1,1,1,1,1,1,1]])
        m=retrieval_metrics(y,ranks)
        self.assertEqual(m['hit_at_1'],0); self.assertEqual(m['hit_at_5'],1)
        self.assertAlmostEqual(m['map_at_10'],sum(i/(i+1) for i in range(1,10))/10)
    def test_pair_balance_and_separation(self):
        texts=[f'caption number {i}' for i in range(12)]
        ids={'train':list(range(8)),'test':list(range(8,12))}; p=make_pairs(ids,texts)
        self.assertTrue((p.groupby(['image_id','label']).size()==1).all())
        self.assertTrue((p.groupby(['text_id','label']).size()==1).all())
        for s,v in ids.items(): self.assertEqual(set(p.query('split == @s').text_id),set(v))


if __name__=='__main__': unittest.main()
