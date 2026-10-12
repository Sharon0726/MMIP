"""Independent saved-output checks: split separation, metrics, probabilities, paired comparisons."""
import json
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from common import ROOT, classification_metrics, save_json, tokens


def main():
    checks=[]
    for task in ['sentiment','vision','caption']:
        split=pd.read_csv(ROOT/f'results/{task}_split.csv')
        key='sha256' if task=='sentiment' else 'filename' if task=='caption' else None
        if key:
            assert split[key].is_unique
        else:
            assert not set(split.loc[split.split=='train','source_id']) & set(split.loc[split.split=='validation','source_id'])
        checks.append(task+' split separation')
    for task,names in [('sentiment',['rnn','lstm']),('vision',['vit','resnet'])]:
        metrics=pd.read_csv(ROOT/f'results/{task}_metrics.csv').set_index('model')
        paired=[]
        for name in names:
            df=pd.read_csv(ROOT/f'results/predictions/{task}_{name}.csv')
            p=df[[c for c in df if c.startswith('p_')]].to_numpy()
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
            assert np.allclose(p.sum(1),1,atol=1e-5)
            assert np.array_equal(p.argmax(1),df.predicted)
            for k,v in classification_metrics(df.true.to_numpy(),p).items():
                assert np.isclose(v,metrics.loc[name,k],atol=1e-6), (task,name,k)
            paired.append(df[['id','true']])
        assert paired[0].equals(paired[1])
        assert len(paired[0])==len(pd.read_csv(ROOT/f'results/{task}_split.csv').query('split == "test"'))
        checks.append(task+' metrics and paired test IDs')
    metric=pd.read_csv(ROOT/'results/caption_metrics.csv').set_index('model')
    paired=[]
    for name in ['gru','lstm']:
        rows=json.loads((ROOT/f'results/predictions/caption_{name}.json').read_text())
        refs=[[tokens(t) for t in r['references']] for r in rows]
        hyps=[tokens(r['generated']) for r in rows]
        for label,weights in [('bleu1',(1,0,0,0)),('bleu4',(.25,)*4)]:
            score=corpus_bleu(refs,hyps,weights=weights)
            assert np.isclose(score,metric.loc[f'ResNet18 + {name.upper()}',label])
        paired.append([r['source_id'] for r in rows])
        assert all(r['generated'] for r in rows)
        times=pd.read_csv(ROOT/f'results/caption_latency_{name}.csv')
        assert (times.milliseconds>0).all()
        assert np.isclose(times.milliseconds.mean(),metric.loc[f'ResNet18 + {name.upper()}','mean_ms'])
    assert paired[0]==paired[1]
    assert len(paired[0])==len(pd.read_csv(ROOT/'results/caption_split.csv').query('split == "test"'))
    checks += ['caption BLEU and paired test IDs','latency measured samples','nonempty generated captions']
    v=pd.read_csv(ROOT/'results/vision_metrics.csv')
    assert (v.backbone_max_abs_change>0).all()
    checks.append('both pretrained backbones really updated')
    save_json('results/verification.json',{'passed':True,'checks':checks})
    print('Verification passed:', '; '.join(checks))


if __name__=='__main__': main()
