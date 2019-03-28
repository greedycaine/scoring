#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np


### Basic Operations ###
# 删除不需要的列
def delFromVardict(df, vardict):
    delete = vardict.loc[vardict['isDel'] == 1, 'new']
    df1 = df.drop(delete.tolist(), axis=1)
    vardict1 = vardict.drop(vardict.index[vardict['new'].isin(delete)].tolist(), axis=0)

    return df1, vardict1


# 对所有列进行重命名，默认重命名为col1, col2, ..., coln
def renameCols(df, vardict, inplace=True):
    
    if inplace:
        df.columns = vardict['new'].tolist()
        return df
    else:
        df1=df.copy()
        df1.columns = vardict['new'].tolist()
        return df1


# 区分离散连续
def getVarTypes(vardict):

    label = vardict.loc[vardict['type'] == 'label', 'new']
    disc = vardict.loc[vardict['type'] == 'disc', 'new']
    cont = vardict.loc[vardict['type'] == 'cont', 'new']
    return list(label)[0], list(disc), list(cont)


# 填充缺失
def fillMissing(df, varDict, filling=-999):
    numCol = varDict.loc[varDict.isNum == 1, 'new']
    charCol = varDict.loc[varDict.isNum == 0, 'new']

    df[numCol] = df[numCol].fillna(filling)
    df[charCol] = df[charCol].fillna(str(filling))

    return df


# 多少不同值，最大占比，最小占比，空值占比
def describe(df):
    tmplist = []
    for i in df.columns:
        # 非空值集
        tmp1 = df.loc[~df[i].isna(), i]
        # 空值集
        tmp2 = df.loc[df[i].isna(), i]
        tmp1rate = tmp1.value_counts() / tmp1.shape[0]

        if tmp2.shape[0] > 0:
            nuni = tmp1.apply(lambda x: str(x)).nunique() + 1
        else:
            nuni = tmp1.apply(lambda x: str(x)).nunique()

        tmp2 = [i,
                df.shape[0],
                nuni,
                #                 max(df[i]),
                #                 min(df[i]),
                #                 mean(df[i]),
                format(nuni / df.shape[0], '.2%'),
                format(max(tmp1rate), '.2%'),
                format(min(tmp1rate), '.2%'),
                format(tmp2.shape[0] / df.shape[0], '.2%')
                ]
        tmplist.append(tmp2)
    df1 = pd.DataFrame(tmplist,
                       columns=['Feature Name',
                                'total',
                                '# of uniques',
                                '% of uniques',
                                'Max % w/n NA',
                                'Min % w/n NA',
                                'Missing rate'])

    return df1