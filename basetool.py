import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2 as c


def describeMissing(df):
    
    return df.isnull().sum()

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

def calcChisq(df):
    
    totaltmpbad=sum(df['bad'])
    totaltmpgood=sum(df['good'])
    total=totaltmpbad+totaltmpgood
    
    ebasebad=totaltmpbad/total
    ebasegood=totaltmpgood/total
    
    Ebase=df.iloc[0,0]/total
    Ebad1=df.iloc[0,0]*ebasebad
    Ebad2=df.iloc[1,0]*ebasebad
    Egood1=df.iloc[0,0]*ebasegood
    Egood2=df.iloc[1,0]*ebasegood

    chisq=(list(df['bad'])[0]-Ebad1)**2/Ebad1+(list(df['good'])[0]-Egood1)**2/Egood1\
            +(list(df['bad'])[1]-Ebad2)**2/Ebad2+(list(df['good'])[1]-Egood2)**2/Egood2
    
    return 0 if pd.isnull(chisq) else chisq

def binByChisq(df,col,y,maxInterval=6,minInterval=2,threshold=None):
    
    if threshold==None:
        threshold=getChiDist()
    
    grouped=df[[col,y]].groupby([col])
    group_to_bin=grouped.count()
    group_to_bin['bad']=grouped.sum()
    group_to_bin['good']=grouped.count()-grouped.sum()

    if is_numeric_dtype(group_to_bin.index)==True:
        
        if df[col].nunique()>200:
            left=list(pd.qcut(df[col], 100, duplicates='drop').apply(lambda x:x.left if isinstance(x,pd.Interval) else x))
            df[col]=left
        
        chisqlist=[]
    
        for i in range(group_to_bin.shape[0]-1):
            tmp=group_to_bin.iloc[i:i+2,:]
            chisqlist.append(calcChisq(tmp))

        group_to_bin['chisq']=chisqlist+[np.inf]
        
        while group_to_bin.shape[0]>minInterval and (group_to_bin.shape[0]>maxInterval or min(group_to_bin['chisq'])<threshold):

            indextobin=list(group_to_bin['chisq']).index(min(group_to_bin['chisq']))

            if indextobin<group_to_bin.shape[0]-1:
                group_to_bin.iloc[indextobin,:]=group_to_bin.iloc[indextobin,:]+group_to_bin.iloc[indextobin+1,:]

                group_to_bin.drop(index=group_to_bin.iloc[indextobin:indextobin+2,:].index[1],axis=0,inplace=True)
                if indextobin==group_to_bin.shape[0]-1:
                    group_to_bin.iloc[indextobin,3]=np.inf
                else:
                    group_to_bin.iloc[indextobin,3]=calcChisq(group_to_bin.iloc[indextobin:indextobin+2,:])
            else:
                continue
        if indextobin>0:
            group_to_bin.iloc[indextobin-1,3]=calcChisq(group_to_bin.iloc[indextobin-1:indextobin+1,:])
        
        cutoff=list(group_to_bin.index)
        cutoff[0]=-np.inf
#         cutoff[len(cutoff)-1]=np.inf
        cutoff.append(np.inf)
    else:
        group_to_bin=group_to_bin.loc[(group_to_bin['bad']/group_to_bin['y']).sort_values().index]

        chisqlist=[]
    
        for i in range(group_to_bin.shape[0]-1):
            tmp=group_to_bin.iloc[i:i+2,:]
            chisqlist.append(calcChisq(tmp))
    
        chisqlist=[0 if pd.isnull(i) else i for i in chisqlist] 
        group_to_bin['chisq']=chisqlist+[np.inf]
         
        while group_to_bin.shape[0]>minInterval and (group_to_bin.shape[0]>maxInterval or min(group_to_bin['chisq'])<threshold):
            indextobin=list(group_to_bin['chisq']).index(min(group_to_bin['chisq']))
            group_to_bin.iloc[indextobin,:]=group_to_bin.iloc[indextobin,:]+group_to_bin.iloc[indextobin+1,:]
            group_to_bin.rename(index={group_to_bin.iloc[indextobin,:].name:
                                       group_to_bin.iloc[indextobin,:].name+'#/#'+group_to_bin.iloc[indextobin+1,:].name},
                                inplace=True)
            group_to_bin.drop([group_to_bin.iloc[indextobin+1,:].name],axis=0,inplace=True)
            if indextobin==group_to_bin.shape[0]-1:
                group_to_bin.iloc[indextobin,3]=np.inf
            else:
                group_to_bin.iloc[indextobin,3]=calcChisq(group_to_bin.iloc[indextobin:indextobin+2,:])
        
            if indextobin>0:
                group_to_bin.iloc[indextobin-1,3]=calcChisq(group_to_bin.iloc[indextobin-1:indextobin+1,:])
        
        cutoff=list(group_to_bin.index)
    return cutoff

def getBinnedCol(df,col,y):

    if is_numeric_dtype(df[col])==True:
        
#         卡方分箱的时候是向下合并，所以左闭右开
        return pd.cut(df[col], binByChisq(df,col,y), right=False)
    else:
        mapdict={}
        for i in binByChisq(df,col,y):
            for j in i.split('#/#'):
                mapdict[j]=i
                
        return df[col].map(mapdict)

    
def bivariate(df, col, label, withIV=True, dealMissing=True):
    df = df.replace(np.nan, '0_missing')
    gb = df.groupby(col, as_index=False)

    total = df.shape[0]
    all = gb.count()
    bad = gb.sum()[label]
    good = (all[label] - bad)

    bitable = pd.DataFrame({col: all[col], 'total': good + bad, 'good': good, 'bad': bad}). \
        replace(0, 0.000001). \
        assign(totalDist=lambda x: x.total / sum(x.total),
               goodDist=lambda x: x.good / sum(x.good),
               badDist=lambda x: x.bad / sum(x.bad),
               goodRate=lambda x: x.good / (x.total),
               badRate=lambda x: x.bad / (x.total)
               ). \
        assign(woe=lambda x: np.log(x.badDist / x.goodDist))

    if withIV:
        bitable['iv'] = (bitable['badDist'] - bitable['goodDist']) * bitable['woe']

    totalIV = sum(bitable['iv'])
    bitable['totalIV']=totalIV

    return bitable, totalIV


def grouped(df,col,y='y'):
    
    df=df.copy(deep=True)
    df.fillna('0_missing',inplace=True)
    
    if df[col].nunique()>200:
        return df.groupby(pd.qcut(df[col],100,duplicates='drop'))[y]
    else:
        return df.groupby([col])[y]


def cal_woe(df, col, y = 'y', nocut=False):
    """
    df:数据集
    col:特征名
    y:样本定义根据的列名（1:正样本，0:负样本）
    """

    if nocut==False:
        group=grouped(df,col,y)
    else:
        group=df.groupby([col])[y]
    # 正样本
    pos_cnt = group.sum()
    # 负样本
    neg_cnt = group.count()-group.sum()
    #所有黑样本
    pos_cnt_total = df[y].sum()
    #所有白样本
    neg_cnt_total = df.shape[0] - pos_cnt_total
    #每组正样本占总体正样本比例
    ppi = (pos_cnt / pos_cnt_total).apply(lambda x: max(0.000001,x))
    #每组负样本占总体负样本比例
    pni = (neg_cnt / neg_cnt_total).apply(lambda x: max(0.000001,x))
  
  #woe
    woe = (ppi / pni).map(lambda x:math.log(x))
  
    return woe, ppi, pni

# 逻辑代码
def cal_iv(df, col, y = 'y' ):
    """
    df:数据集
    col:特征名
    y:样本定义根据的列名（1:正样本，0:负样本）
    """
    # 获取woe、ppi、pni
    woe, ppi, pni = cal_woe(df, col, y)
    # 计算特征每个分箱的iv值
    ivi = (ppi - pni) * woe
    # 返回该特征的iv值
    return ivi.sum()


def cal_KS(df, col, y = 'y', nocut=False):
    
    """
    df:数据集
    col:特征名
    y:样本定义根据的列名（1:正样本，0:负样本）
    """
    
    if nocut==False:
        group=grouped(df,col,y)
    else:
        group=df.groupby([col])[y]
    # 正样本
    pos_cnt = group.sum()
    # 负样本
    neg_cnt = group.count()-group.sum()
    #所有黑样本
    pos_cnt_total = df[y].sum()
    #所有白样本
    neg_cnt_total = df.shape[0] - pos_cnt_total
    
    return max(abs(pos_cnt/pos_cnt_total-neg_cnt/neg_cnt_total))


df=pd.read_csv('../data/gc.csv')
df['y']=df['Risk'].apply(lambda x: 1 if x=='bad' else 0)
for i in df:
    df[i+'1']=getBinnedCol(df,i,'y')
    print(bivariate(df,i+'1','y'))
