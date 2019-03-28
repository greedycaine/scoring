# Model Evaluating
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import itertools

from .discretization import equalDepthBinning,manuallyBin
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

### Evaluation ###
##################
# roc
# 从高到低，依次将“Score”值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本。
# 每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。
def plotROC(testy, pred_prob, title='ROC', get_parameters=False):
    
    fpr,tpr,threshold=metrics.roc_curve(testy,pred_prob)
    roc_auc=metrics.auc(fpr,tpr)
    fig=plt.figure()
    plt.plot(fpr,tpr,'b',label='ROC = %0.4f'% roc_auc)
    plt.plot([0,1],[0,1],'--',label='RANDOM = %0.4f'% 0.5)
    plt.legend(loc='lower right')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    
    if get_parameters:
        return roc_auc,fpr,tpr,threshold
    
def plotKS(testy, pred_prob, group=30, title='KS', get_parameters=False):
    
    fpr,tpr,threshold=metrics.roc_curve(testy,pred_prob)
    fpr1=np.quantile(fpr,np.linspace(0,1,group+1))
    tpr1=np.quantile(tpr,np.linspace(0,1,group+1))
    ks=tpr1-fpr1

    fig=plt.figure()
    plt.plot(ks,color='blue',linestyle='--',label='KS')
    plt.axvline(pd.Series(ks).idxmax(),color='blue',linestyle=':',label='Max KS')  
    plt.plot(tpr1,color='green',label='TPR')
    plt.plot(fpr1,color='red',label='FPR')
    plt.legend(loc='right')
    plt.xlabel('Decili')
    plt.ylabel('Cumulative Rate')
    plt.title('{}\n(KS={:.4F})'.format(title,max(ks)))
    plt.show()
    
    if get_parameters:
        return max(ks)

def plotCM(cm, classes,
           normalize=False,
           title='Confusion matrix',
           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# Calculating PSI(Population Stability Index)
def calcPSI(df1, df2, col, n=10, getValue=False):
    s1, bins = equalDepthBinning(df1, col, n)
    s2 = manuallyBin(df2, col, vartype='cont', cutoff=sorted(bins.unique()) + [np.inf])
    s1dist = pd.DataFrame({'s1dist': s1.value_counts() / s1.shape[0]})
    s2dist = pd.DataFrame({'s2dist': s2.value_counts() / s2.shape[0]})
    psi = s1dist.join(s2dist).sort_index()
    psi['psi'] = (psi.s2dist - psi.s1dist) * np.log(psi.s2dist / psi.s1dist)
    psi['totalpsi'] = sum(psi.psi)

    if getValue:
        return psi, sum(psi.psi)
    else:
        return psi


def plotCorr(woedf):

    corr = woedf.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(woedf.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(woedf.columns)
    ax.set_yticklabels(woedf.columns)
    plt.show()


def calcVIF(df):

    return vif(df.values,1)