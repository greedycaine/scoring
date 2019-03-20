# Model Evaluating
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import itertools



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