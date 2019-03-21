import pandas as pd
import numpy as np
from .cleanning import delFromVardict

def calcWOE(allGoodCnt, allBadCnt, eachGoodCnt, eachBadCnt):

    woe = np.log((eachGoodCnt / eachBadCnt) / (allGoodCnt / allBadCnt))

    return woe


def calcIV(allGoodCnt, allBadCnt, eachGoodCnt, eachBadCnt):
    # calcIV(allGoodCnt, allBadCnt, eachGoodCnt, eachBadCnt, label='DEFAULT')
    woe = calcWOE(allGoodCnt, allBadCnt, eachGoodCnt, eachBadCnt)
    ivcolumn = (eachGoodCnt / allGoodCnt - eachBadCnt / allBadCnt) * woe
    iv = sum(ivcolumn)

    return ivcolumn, iv


def bivariate(df, col, label, withIV=True, missingvalue='missing', dealMissing=True):
    df = df.replace(np.nan, missingvalue)
    gb = df.groupby(col, as_index=False)

    total = df.shape[0]
    all = gb.count()
    bad = gb.sum()[label]
    good = (all[label] - bad)

    bitable = pd.DataFrame({col: all[col], 'total': good + bad, 'good': good, 'bad': bad}). \
        replace(0, 0.001). \
        assign(totalDist=lambda x: x.total / sum(x.total),
               goodDist=lambda x: x.bad / sum(x.bad),
               badDist=lambda x: x.good / sum(x.good),
               goodRate=lambda x: x.good / (x.total),
               badRate=lambda x: x.bad / (x.total)
               ). \
        assign(woe=lambda x: np.log(x.badDist / x.goodDist))

    if withIV:
        bitable['iv'] = (bitable['badDist'] - bitable['goodDist']) * bitable['woe']

    totalIV = sum(bitable['iv'])

    return bitable, totalIV


# return a dictionary of results of bivariate analysis
def getBiDict(df, label, getiv=False):
    bidict = {}
    ivdict = {}
    for i in df.drop([label], axis=1).columns:
        tmp = bivariate(df, i, label)
        bidict[i] = tmp[0]
        ivdict[i] = tmp[1]
        
    ivtable=pd.DataFrame(list(ivdict.items()),columns=['feature','iv']).sort_values('iv',ascending=False)    
    
    if getiv:
        return bidict,ivtable
    else:
        return bidict
    
    
    
### Transformation ###
######################
# 把woe table转换成字典格式, 生成新的df1包含全部woe以及label
def mapWOE(df, bidict, label, missingvalue='missing'):
    
    df=df.replace(np.nan,missingvalue)
    df1=pd.DataFrame(df[label])
    for i in bidict:
        tmp=bidict[i]
        tmpdict=pd.Series(tmp.woe.values,index=tmp[i]).to_dict()
        tmplist=[]
        for j in df[i]:
            tmplist.append(tmpdict[j])
        df1[i]=pd.Series(tmplist)
    
    return df1



# 默认过滤iv值小于0.02的特征，
# return a dictionary of {feature,iv} pairs, and a iv-filtered list of feature
def ivTable(bidict,threshold=0.02):

    ivdict={}
    for i in list(bidict.keys()):
        ivdict[i]=sum(bidict[i]['iv'])
    ivtable=pd.DataFrame(list(ivdict.items()),
                         columns=['Feature','iv']).sort_values('iv',ascending=False)
    
    ivtable['isKept']=ivtable['iv'].apply(lambda x: 'Y' if x>threshold else 'N')
    
    return ivtable.reset_index(drop=True)


def featureFilter(df,vd,bidict,ivtable):
    
    bidict1=bidict.copy()
    for i in ivtable.loc[ivtable['isKept']=='N','Feature']:
        vd.loc[vd['new']==i,'isDel']=1
        bidict1.pop(i)
    df1,vd1=delFromVardict(df,vd)

    return df1,vd1,bidict1