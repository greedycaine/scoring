import pandas as pd
import numpy as np

from .cleanning import getVarTypes
from .bivariate import bivariate
from scipy.stats import chi2 as c


# Splitting Dataframe by expect-train ratio or size
def splitDF(df, size=None, ratio=None, random_state=None):
    if size == ratio == None:
        ratio = 0.5
        train = df.sample(frac=ratio, random_state=random_state)
        test = df.loc[~df.index.isin(train.index)]
    elif ratio == None and size > 0:
        print('size')
        train = df.sample(n=size, random_state=random_state)
        test = df.loc[~df.index.isin(train.index)]
    elif size == None and ratio >= 0 and ratio <= 1:
        train = df.sample(frac=ratio, random_state=random_state)
        test = df.loc[~df.index.isin(train.index)]
    else:
        print("Incorrect Inputs.")

    return train.reset_index(drop=True), test.reset_index(drop=True)


# unsupervised binning
def equalDepthBinning(df,col,n=None):
    
    if n==None:
        n=getBinNum(df,col)
        
    interval=pd.qcut(df[col],n,duplicates='drop')
    minimum=interval.value_counts().sort_index().index[0]
    maximum=interval.value_counts().sort_index().index[interval.nunique()-1]

    interval=interval.replace(minimum,pd.Interval(left=-np.Inf,right=minimum.right))
    interval=interval.replace(maximum,pd.Interval(left=maximum.left,right=np.Inf))
    
#     left=interval.apply(lambda x:x.left)
    left=interval.apply(lambda x:x.left if isinstance(x,pd.Interval) else x)

    return interval,left
def equalWidthBinning(df,col,n=None):
    
    if n==None:
        n=getBinNum(df,col)

    interval=pd.cut(df[col],n,duplicates='drop')
    minimum=interval.value_counts().sort_index().index[0]
    maximum=interval.value_counts().sort_index().index[interval.nunique()-1]
    
    interval=interval.replace(minimum,pd.Interval(left=-np.Inf,right=minimum.right))
    interval=interval.replace(maximum,pd.Interval(left=maximum.left,right=np.Inf))

#     left=interval.apply(lambda x:x.left)
    left=interval.apply(lambda x:x.left if isinstance(x,pd.Interval) else x)
    
    return interval,left
def applyEDB(df, collist, n=10):
    
    tmp=df.copy()
    for i in collist:
        tmp[i]=equalDepthBinning(tmp,i,n)
    
    return tmp
def applyEWB(df, collist, n=10):
    
    tmp=df.copy()
    for i in collist:
        tmp[i]=equalDepthBinning(tmp,i,n)
    
    return tmp

# supervised binning
## Chimerge
def getBinNum(df,col):

    n=df[col].nunique()
    if n>100:
        x=10**np.floor(np.log10(n))
        y=np.floor(n/x)*x
        z=max(y,10000)/100
        return int(z)
    else:
        return None




def getChiDist(dof=1, sl=0.1):
    '''
    根据自由度和置信度得到卡方分布和阈值
    dfree:自由度k= (行数-1)*(列数-1)，默认为4     #问题，自由度k,如何来确定？
    cf:显著性水平，默认10%
    '''
    percents = [ 0.95, 0.90, 0.5,0.1, 0.05, 0.025, 0.01, 0.005]
    df = pd.DataFrame(np.array([c.isf(percents, df=i) for i in range(1, 30)]))
    df.columns = percents
    df.index = df.index+1
    # 显示小数点后面数字
    pd.set_option('precision', 3)
    return df.loc[dof, sl]


def calc_chi2(dfsumm):
    
    chi2_result=[]
    for i in np.arange(0,dfsumm.shape[0]-1):

        Ni=dfsumm.bad[i]+dfsumm.good[i]
        Ni1=dfsumm.bad[i+1]+dfsumm.good[i+1]
        N=Ni+Ni1

        chi2=0
        for j in ['bad','good']:

            Cj=dfsumm[j][i]+dfsumm[j][i+1]
            Eij=Ni*Cj/N
            Ei1j=Ni1*Cj/N

            chi2=chi2+(dfsumm[j][i]-Eij)**2/Eij+(dfsumm[j][i+1]-Ei1j)**2/Ei1j

        if np.isnan(chi2):
            chi2_result.append(0)
        else:
            chi2_result.append(chi2)
    
    return chi2_result


def mergeContByIndex(dfsumm,index):
        
    if index+1>=dfsumm.shape[0]:
        index=index-1
    
    dfsumm.iloc[index,1:5]=dfsumm.iloc[index,1:5]+dfsumm.iloc[index+1,1:5]
    dfsumm.loc[dfsumm.index==index,'badRate']=dfsumm.loc[dfsumm.index==index,'bad']/dfsumm.loc[dfsumm.index==index,'total']
    dfsumm=dfsumm.drop(index+1,axis=0).reset_index(drop=True)
    
    return dfsumm

def mergeDiscByIndex(dfsumm,col,index):
    
    if index+1>=dfsumm.shape[0]:
        index=index-1
    
    dfsumm.iloc[index,1:5]=dfsumm.iloc[index,1:5]+dfsumm.iloc[index+1,1:5]
    dfsumm.loc[dfsumm.index==index,'badRate']=dfsumm.loc[dfsumm.index==index,'bad']/dfsumm.loc[dfsumm.index==index,'total']
    dfsumm.iloc[index,0].extend(dfsumm.iloc[index+1,0])
    dfsumm=dfsumm.drop(index+1,axis=0).reset_index(drop=True)
    
    return dfsumm


def binByChi2(df,col,label,vartype,
              maxIntervals=6,minIntervals=2,
              threshold=False,
              minIntPect=0.05,
              dof=1,sl=0.1,
              n=None,inPercentum=True,getCutOff=False):
    
    # if n==None:
    #     n=getBinNum(df,col)

    if n==None:
        n=min(df[col].nunique(),100)
    else:
        n=min(n,df[col].nunique(),100)
    
    tmp=df[[col,label]].copy()
    if vartype=='cont' and n!=None:
        print("Equal Depth Binning is required, number of bins is:", n)
        tmp[col]=equalDepthBinning(tmp,col,n)[1]
        
    total=tmp.groupby(col).count()
    bad=tmp.groupby(col).sum()
    good=total-bad
    badr=bad/total
    occr=total/sum(total[label])
    
    if threshold==False:
        threshold=getChiDist(dof,sl)
        
    hasMissing=True if tmp.loc[tmp[col].isna(),label].shape[0] else False
    
    tmpsumm=pd.DataFrame({'total':total[label],
                          'bad':bad[label],
                          'good':good[label]}).\
                    replace(0,0.001).\
                    assign(occRate=lambda x:x.total/sum(x.total),
                           badRate=lambda x:x.bad/x.total)

    if vartype=='cont':
        tmpsumm=tmpsumm.sort_values(col).reset_index()
    elif vartype=='disc':
        tmpsumm=tmpsumm.sort_values('badRate').reset_index()
        tmpsumm[col]=[[i] for i in tmpsumm[col]]

    tmpsumm['chi2']=calc_chi2(tmpsumm)+[np.inf]    
    
    while (tmpsumm.shape[0]>minIntervals) and (tmpsumm.shape[0]>maxIntervals or \
                                               min(tmpsumm['chi2'])<threshold or \
                                               min(tmpsumm.occRate)<minIntPect):
        # first, check the threshold
        if min(tmpsumm['chi2'])<threshold or tmpsumm.shape[0]>maxIntervals:

            # 需要确定是用最小卡方值对比阈值，再将所有最小卡方值的合并；
            # 还是确定出小于阈值的所有值，在一次按从小到大合并：
            ## 即list1为小于阈值的值合集，list2为各个list1值中的索引合集，for i in list1: for j in list2: mergebin
            merge_idx=tmpsumm[tmpsumm['chi2']==min(tmpsumm['chi2'])].index[0]
            if vartype=='cont':
                tmpsumm=mergeContByIndex(tmpsumm,merge_idx)
            elif vartype=='disc':
                tmpsumm=mergeDiscByIndex(tmpsumm,col,merge_idx)

        elif min(tmpsumm.occRate)<minIntPect:

            merge_idx=tmpsumm[tmpsumm.occRate==min(tmpsumm.occRate)].index[0]
            if vartype=='cont':
                tmpsumm=mergeContByIndex(tmpsumm,merge_idx)
            elif vartype=='disc':
                tmpsumm=mergeDiscByIndex(tmpsumm,col,merge_idx)

        tmpsumm['chi2']=calc_chi2(tmpsumm)+[np.inf]
    
    if vartype=='cont':
        cutoff=tmpsumm[col].tolist()+[np.inf]
        for i in np.arange(tmpsumm.shape[0]-1):
            tmpsumm.loc[i,col]=pd.Interval(left=tmpsumm.loc[i,col],
                                              right=tmpsumm.loc[i+1,col],
                                              closed='right')
        tmpsumm.loc[tmpsumm.shape[0]-1,col]=pd.Interval(left=tmpsumm.loc[tmpsumm.shape[0]-1,col],
                                                           right=np.inf,
                                                           closed='right')
    elif vartype=='disc':
        cutoff=tmpsumm[col].tolist()

        
    if hasMissing==True:
        missingdf=tmp.loc[tmp[col].isna(),label]
        mtotal=missingdf.count()
        mbad=missingdf.sum()
        mgood=mtotal-mbad
        moccRate=mtotal/tmp.shape[0]
        mbadRate=mbad/mtotal
        mchi2=0
        
        tmpsumm=tmpsumm.append({'bins':'missing',
                                col:'missing',
                                'total':mtotal,
                                'bad':mbad,
                                'good':mgood,
                                'occRate':moccRate,
                                'badRate':mbadRate,
                                'chi2':mchi2},ignore_index=True)
                
        tmpsumm['chi2']=calc_chi2(tmpsumm)+[np.inf]
    
    tmpsumm['bad']=tmpsumm['bad'].apply(lambda x: int(x))
    tmpsumm['good']=tmpsumm['good'].apply(lambda x: int(x))
    tmpsumm['total']=tmpsumm['total'].apply(lambda x: int(x))
    
    if inPercentum==True:
        tmpsumm['occRate']=tmpsumm['occRate'].apply(lambda x: format(x,'.2%'))
        tmpsumm['badRate']=tmpsumm['badRate'].apply(lambda x: format(x,'.2%'))

    if getCutOff==True:
        return [tmpsumm[[col, 'total', 'bad', 'good', 'occRate', 'badRate', 'chi2']],cutoff]
    else:
        return tmpsumm[[col, 'total', 'bad', 'good', 'occRate', 'badRate', 'chi2']]


# manually binning
def manuallyBin(df, col, label, vartype, cutoff, bi=False):

    print('Binning data by cutoff:', cutoff)
    if vartype == 'cont':
        if bi:
            return bivariate(pd.DataFrame({label: df[label], col: pd.cut(df[col], cutoff)}), col, label)[0]
        else:
            return pd.cut(df[col], cutoff)
    elif vartype == 'disc':
        res = []
        found = False
        for i in df[col].replace(np.nan, 'missing'):
            if i == 'missing':
                found = True
                res.append(np.nan)
            else:
                for j in np.arange(len(cutoff)):
                    if i in cutoff[j]:
                        found = True
                        res.append(str(cutoff[j]))
            if found == False:
                res.append('others')
            found = False

        if bi:
            return bivariate(pd.DataFrame({label: df[label], col: res}), col, label)[0]
        else:
            return res


# def binData(df,vardict,method='chimerge'):
def binData(df, vardict, altdict=None, method='chimerge'):

    tmp=df.copy()
    label, disc, cont = getVarTypes(vardict)
    cutoffdict={}

    if method=='chimerge':
        print("#########################################")
        print("####It's using Chi-Merge algorithm...####")
        print("#########################################")



        for i in cont:
            print('\nDoing continous feature:',i)
            if altdict!=None and i in altdict.keys():
                cutoffdict[i]=altdict[i]
                tmp[i] = manuallyBin(tmp,i,label,'cont',altdict[i])
            else:
                cutoffdict[i]=binByChi2(tmp, i, label, 'cont', getCutOff=True)[1]
                tmp[i] = manuallyBin(tmp, i, label, 'cont', cutoffdict[i])

        for i in disc:
            print('\nDoing discrete feature:',i)
            if altdict!=None and i in altdict.keys():
                cutoffdict[i]=altdict[i]
                tmp[i] = manuallyBin(tmp,i,label,'cont',altdict[i])
            else:
                cutoffdict[i]=binByChi2(tmp, i, label, 'disc', getCutOff=True)[1]
                tmp[i] = manuallyBin(tmp, i, label, 'disc', cutoffdict[i])

        print('\nFinished')
        return tmp,cutoffdict
    elif method=='CART':
        print("It's using CART algorithm...")

        # TBD
        print('\nFinished')
        return tmp,cutoffdict
    else:
        print('Incorrect method chosen, original dataframe is returned.')
        return tmp,cutoffdict