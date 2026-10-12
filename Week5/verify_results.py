"""Independently recompute reported metrics and audit data pairing from saved outputs."""
import hashlib
import json
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from common import ROOT,classification_metrics,tokens,save_json


def equal(a,b): assert np.isclose(a,b,rtol=1e-5,atol=1e-6),(a,b)


def main():
    checks=[]
    df=pd.read_csv(ROOT/'results/retrieval_split.csv')
    assert len(df)==32000 and not df.duplicated(['source','source_id']).any()
    assert df.groupby('split').size().to_dict()=={'gallery':10000,'query':10000,'train':10000,'validation':2000}
    gallery=df.query('split == "gallery"').set_index('source_id').label
    query=df.query('split == "query"').set_index('source_id').label
    rows=pd.read_csv(ROOT/'results/retrieval_metrics.csv').set_index('model')
    for tag in ['resnet','vit']:
        p=np.load(ROOT/f'results/predictions/retrieval_{tag}.npz')
        assert p['gallery_ids'].shape==(10000,10)
        assert np.array_equal(query.loc[p['query_ids']].to_numpy(),p['query_labels'])
        assert np.array_equal(gallery.loc[p['gallery_ids'].ravel()].to_numpy().reshape(-1,10),p['neighbor_labels'])
        assert (np.diff(p['similarities'],axis=1)<=1e-6).all() and np.isfinite(p['similarities']).all()
        rel=(p['query_labels'][:,None]==p['neighbor_labels']).astype(float)
        for k in [1,5,10]: equal(rows.loc[tag,f'hit_at_{k}'],rel[:,:k].any(1).mean())
        equal(rows.loc[tag,'precision_at_5'],rel[:,:5].mean())
        ap=np.array([sum(r[:j+1].mean()*r[j] for j in range(10))/10 for r in rel])
        equal(rows.loc[tag,'map_at_10'],ap.mean())
        assert rows.loc[tag,'backbone_max_abs_change']>0
        latency=pd.read_csv(ROOT/f'results/retrieval_latency_{tag}.csv')
        assert len(latency)==100 and (latency.milliseconds>0).all()
        equal(rows.loc[tag,'mean_query_ms'],latency.milliseconds.mean())
    checks.append('Retrieval partitions, qualified IDs, ranked labels, Hit@K/P@5/mAP@10, latency, backbone update')
    split=pd.read_csv(ROOT/'results/flickr_split.csv')
    old=pd.read_csv(ROOT/'baseline_week4/caption_split.csv'); pd.testing.assert_frame_equal(split,old)
    assert split.filename.is_unique and split.source_id.is_unique
    pairs=pd.read_csv(ROOT/'results/multimodal_pairs.csv')
    for s in ['train','validation','test']:
        p=pairs.query('split == @s'); valid=set(split.query('split == @s').source_id)
        assert set(p.image_id)==valid and set(p.text_id)==valid
        assert (p.loc[p.label==1,'image_id']==p.loc[p.label==1,'text_id']).all()
        assert (p.loc[p.label==0,'image_id']!=p.loc[p.label==0,'text_id']).all()
        assert (p.groupby(['image_id','label']).size()==1).all() and (p.groupby(['text_id','label']).size()==1).all()
    rows=pd.read_csv(ROOT/'results/multimodal_metrics.csv').set_index('model')
    expected=pairs.query('split == "test"').reset_index(drop=True)
    for tag in rows.index:
        p=pd.read_csv(ROOT/f'results/predictions/multimodal_{tag}.csv')
        pd.testing.assert_frame_equal(p[expected.columns],expected)
        probs=p[['p_0','p_1']].to_numpy(); assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
        assert np.array_equal(probs.argmax(1),p.prediction)
        for key,value in classification_metrics(p.label.to_numpy(),probs).items(): equal(rows.loc[tag,key],value)
    checks.append('Multimodal within-split pairing, per-input label balance, common test IDs, probability metrics')
    expected_ids=split.query('split == "test"').source_id.to_list()
    metrics=pd.read_csv(ROOT/'results/caption_comparison.csv').set_index('model'); first=None
    for tag in metrics.index:
        path=ROOT/f'results/predictions/caption_{tag}.json' if tag.startswith('clip') else ROOT/f'baseline_week4/caption_{tag.split("_")[1]}.json'
        rows=json.loads(path.read_text(encoding='utf-8')); assert [r['source_id'] for r in rows]==expected_ids
        assert all(r['generated'].strip() for r in rows)
        ref=[[tokens(t) for t in r['references']] for r in rows]; hyp=[tokens(r['generated']) for r in rows]
        if first is None: first=ref
        else: assert ref==first
        equal(metrics.loc[tag,'bleu1'],corpus_bleu(ref,hyp,weights=(1,0,0,0)))
        equal(metrics.loc[tag,'bleu4'],corpus_bleu(ref,hyp,weights=(.25,)*4))
        b=pd.read_csv(ROOT/f'results/predictions/bertscore_{tag}.csv'); assert b.source_id.to_list()==expected_ids
        assert np.isfinite(b[['precision','recall','f1']]).all().all()
        for key in ['precision','recall','f1']: equal(metrics.loc[tag,f'bertscore_{key}'],b[key].mean())
    checks.append('Exact Week4 split/references, nonempty captions, corpus BLEU recomputation and BERTScore per-image mean')
    # BERTScore neural token matching was computed on GPU; this lightweight check verifies aggregation, not a second model inference.
    source=json.loads((ROOT/'results/source_sha256.json').read_text())
    for name,sha in source.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    checks.append('Executed training source SHA-256 matches delivered source')
    save_json('results/verification.json',{'passed':True,'checks':checks,'bertscore_check':'per-image output aggregation; no CPU rerun of contextual model'})
    print('VERIFIED:', '; '.join(checks),flush=True)


if __name__=='__main__': main()
