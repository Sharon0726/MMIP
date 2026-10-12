"""Plot real histories and test metrics without fabricating results."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,confusion_matrix,ConfusionMatrixDisplay
from common import ROOT


def save(fig,name):
    fig.tight_layout(); fig.savefig(ROOT/f'results/figures/{name}.png',dpi=150); plt.close(fig)


def main():
    for task,tags,ys in [('retrieval',['resnet','vit'],['train_loss','validation_hit_at_1']),
                         ('multimodal',['resnet_fusion','vit_fusion'],['train_loss','val_macro_f1']),
                         ('caption',['clip_gru','clip_lstm'],['train_loss','val_loss'])]:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for tag in tags:
            df=pd.read_csv(ROOT/f'results/history/{task}_{tag}.csv')
            for ax,y in zip(axes,ys): ax.plot(df.epoch,df[y],label=tag); ax.set(xlabel='Epoch',ylabel=y); ax.grid(alpha=.2); ax.legend()
        save(fig,f'{task}_learning')
    retrieval=pd.read_csv(ROOT/'results/retrieval_metrics.csv').set_index('model')
    fig,ax=plt.subplots(figsize=(9,4)); retrieval[['hit_at_1','hit_at_5','precision_at_5','map_at_10']].plot.bar(ax=ax)
    ax.set(ylabel='Score (0-1)',ylim=(0,1),title='Semantic image retrieval on 10,000 held-out queries'); ax.tick_params(axis='x',rotation=0)
    save(fig,'retrieval_comparison')
    fusion=pd.read_csv(ROOT/'results/multimodal_metrics.csv').set_index('model')
    fig,ax=plt.subplots(figsize=(10,4)); fusion[['accuracy','macro_f1','auc']].plot.bar(ax=ax)
    ax.set(ylabel='Score (0-1)',ylim=(0,1),title='Original-pair classification and unimodal controls'); ax.tick_params(axis='x',rotation=12)
    save(fig,'multimodal_comparison')
    fig,ax=plt.subplots(figsize=(6,5))
    for tag in ['resnet_fusion','vit_fusion']:
        df=pd.read_csv(ROOT/f'results/predictions/multimodal_{tag}.csv'); fpr,tpr,_=roc_curve(df.label,df.p_1)
        ax.plot(fpr,tpr,label=tag)
        cmfig,cmax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(confusion_matrix(df.label,df.prediction),display_labels=['permuted','original']).plot(ax=cmax,colorbar=False)
        save(cmfig,f'multimodal_confusion_{tag}')
    ax.plot([0,1],[0,1],'k--',alpha=.5); ax.legend(); ax.set(xlabel='False positive rate',ylabel='True positive rate')
    save(fig,'multimodal_roc')
    caption=pd.read_csv(ROOT/'results/caption_comparison.csv').sort_values(['week','model']).set_index('model')
    fig,axes=plt.subplots(1,2,figsize=(12,4)); caption[['bleu1','bleu4']].plot.bar(ax=axes[0]); caption[['bertscore_f1']].plot.bar(ax=axes[1],color='#287d8e')
    for ax in axes: ax.set(ylim=(0,1),ylabel='Score (0-1)'); ax.tick_params(axis='x',rotation=15)
    axes[0].set_title('Corpus BLEU: all five references'); axes[1].set_title('BERTScore F1: best of five references, unrescaled')
    save(fig,'caption_comparison')


if __name__=='__main__': main()
