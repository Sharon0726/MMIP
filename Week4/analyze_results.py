"""Create figures from saved predictions (no training or fabricated values)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from common import ROOT


def main():
    for task,names in [('sentiment',['rnn','lstm']),('vision',['vit','resnet'])]:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for name in names:
            history=pd.read_csv(ROOT/f'results/history/{task}_{name}.csv')
            axes[0].plot(history.epoch,history.train_loss,label=name)
            metric='val_macro_f1'
            axes[1].plot(history.epoch,history[metric],label=name)
        axes[0].set_title(f'{task}: training loss'); axes[1].set_title('Validation Macro-F1')
        for ax in axes: ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(ROOT/f'results/figures/{task}_learning.png',dpi=160); plt.close(fig)
        for name in names:
            df=pd.read_csv(ROOT/f'results/predictions/{task}_{name}.csv')
            fig,ax=plt.subplots(figsize=(7,6))
            ConfusionMatrixDisplay(confusion_matrix(df.true,df.predicted)).plot(ax=ax,cmap='Blues',colorbar=False)
            ax.set_title(f'{task} / {name} / test'); fig.tight_layout()
            fig.savefig(ROOT/f'results/figures/{task}_{name}_confusion.png',dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,6))
        for name in names:
            df=pd.read_csv(ROOT/f'results/predictions/{task}_{name}.csv')
            if task=='sentiment':
                fpr,tpr,_=roc_curve(df.true,df.p_1)
                ax.plot(fpr,tpr,label=f'{name}: AUC={auc(fpr,tpr):.4f}')
            else:
                grid=np.linspace(0,1,1001); curves=[]
                for k in range(10):
                    fpr,tpr,_=roc_curve((df.true==k).astype(int),df[f'p_{k}'])
                    curves.append(np.interp(grid,fpr,tpr))
                score=pd.read_csv(ROOT/'results/vision_metrics.csv').set_index('model').loc[name,'auc']
                ax.plot(grid,np.mean(curves,axis=0),label=f'{name}: Macro-AUC={score:.4f}')
        ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{task}: test ROC')
        ax.legend(); fig.tight_layout(); fig.savefig(ROOT/f'results/figures/{task}_roc.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ['gru','lstm']:
        df=pd.read_csv(ROOT/f'results/history/caption_{name}.csv')
        axes[0].plot(df.epoch,df.train_loss,label=name); axes[1].plot(df.epoch,df.val_loss,label=name)
    axes[0].set_title('Caption: training token cross-entropy'); axes[1].set_title('Caption: validation token cross-entropy')
    for ax in axes: ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT/'results/figures/caption_learning.png',dpi=160); plt.close(fig)
    df=pd.read_csv(ROOT/'results/caption_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(df.model,df.bleu4,color=['#247BA0','#E07A5F']); axes[0].set_title('Test corpus BLEU-4 (0-1)')
    axes[1].bar(df.model,df.mean_ms,color=['#247BA0','#E07A5F']); axes[1].set_title('Mean single-image latency (ms)')
    fig.tight_layout(); fig.savefig(ROOT/'results/figures/caption_comparison.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
