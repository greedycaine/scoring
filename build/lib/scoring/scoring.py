import pandas as pd
import numpy as np


def calcAB(odds=1/19, pdo=50, p0=600):
    
#     odds=1/(1-p) where p is the bad rate
#     score equation is: score=A+B*ln(odds)
#     hence, let p0 be the base score, we can have:
#     p0=A+B*ln(odds)       (1)
#     p0+pdo=A+B*ln(2odds)  (2)
    
#     solve equations above, we can have:
#     B=pdo/ln(2)           (1)
#     A=p0-Bln(odds)        (2)
    b=pdo/np.log(2)
    a=p0+b*np.log(odds)

    return [a,b]


def calcBinScore(df, bidict, b, coef):
    n = 0

    for i in df.columns:
        bidict[i]['score'] = round(-b * coef[0][n] * bidict[i]['woe'])
        n = n + 1
    return bidict

def mapScore(Xdf,Ydf,bidict,label):
    
    newdf=pd.DataFrame(Ydf)
    for i in bidict:
        tmp=bidict[i]
        tmpdict=pd.Series(tmp.score.values,index=tmp['woe']).to_dict()
        tmplist=[]
        for j in Xdf[i]:
            tmplist.append(tmpdict[j])
        newdf[i]=pd.Series(tmplist)
        
    return newdf


def calcScore(df, model_intercept, a, b, label):
    basescore = round((a - b * model_intercept)[0])
    df['score'] = basescore + df.sum(axis=1) - df[label]

    return [df, basescore]

def scoring(Xdf,Ydf,label,m,bidict):
    
    a,b=calcAB()
    bidict=calcBinScore(Xdf,bidict,calcAB()[1],m.coef_)
    tmp=mapScore(Xdf,Ydf,bidict,label)
    tmp,basescore=calcScore(tmp, m.intercept_, a, b, label)
    
    return [tmp,basescore]


def toScorecard(bidict, basescore, path='./'):
    scorecard = pd.DataFrame(columns=['feature', 'bins',
                                      'total', 'good', 'bad',
                                      'totalDist', 'goodDist', 'badDist',
                                      'goodRate', 'badRate',
                                      'woe', 'iv', 'totalIV',
                                      'score'])

    scorecard = scorecard.append({'feature': 'basesocre', 'score': basescore}, ignore_index=True)

    for i in bidict:
        tmp = bidict[i].rename(columns={i: 'bins'})
        tmp['feature'] = i
        scorecard = scorecard.append(tmp, sort=False)

    scorecard.to_excel(path + 'scorecard.xlsx', index=False)