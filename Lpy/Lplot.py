# coding:utf-8
""" This module is for Proteome plotting """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import NaN
import numpy as np


def heatmapC(data, filename, method='complete', metric='euclidean', row_cluster=True, col_cluster=False):
    """
    :param data:
    :param filename:
    :param method:
    :param metric:
    :param row_cluster:
    :param col_cluster:
    :return: file by pdf
    """
    fig = plt.figure(figsize=(50, 50))
    ax = plt.subplot(111)
    ax.tick_params(labelsize=20)
    sns.clustermap(data=data,
                   method=method,
                   metric=metric,
                   row_cluster=row_cluster,
                   col_cluster=col_cluster,
                   cmap='RdYlBu',
                   col_colors=None,
                   cbar_pos=(0.03, 0.76, 0.03, 0.18),
                   yticklabels=False,
                   xticklabels=True
                   )
    plt.savefig('../test/' + filename + '.pdf', dpi=200)
    plt.show()



def volcano(df,group1:list,group2:list,fc=1,p_value=0.05,str1="grp1",str2="grp2",pair=False,adjust=True):
    """
    # TODO:param,test 'pair',simplify
    """
    list = group1 + group2
    columns = df.columns[list]
    df = pd.DataFrame(df, columns=columns).T
    df = df.dropna(axis=1,how='all')
    pd.set_option('mode.chained_assignment', None)
    df['index'] = df.index
    df['index'][group1] = 'grp1'
    df['index'][group2] = 'grp2'
    df.index = df['index']
    df = df.drop('index', axis=1)

    data = df.applymap(lambda x: 2 ** x if type(x) == float else np.nan)
    m = np.nanmin(data.min().values) * 0.8
    data.fillna(m, inplace=True)
    dff = data.T
    dff.columns=columns
    data = data.groupby(data.index).agg(np.mean).T
    dff['fd'] = np.log2(data['grp1'] / data['grp2'])
    m = np.nanmin(df.min().values) * 0.8
    df.fillna(m, inplace=True)
    from scipy import stats
    if pair:
        x = stats.ttest_rel(df[df.index == 'grp1'], df[df.index == 'grp2'])
    else:
        x = stats.ttest_ind(df[df.index == 'grp1'], df[df.index == 'grp2'], equal_var=False)
    dff['sig']='normal'
    dff['p_value'] = x.pvalue

    try:
        from rpy2.robjects.packages import importr
        from rpy2.robjects.vectors import FloatVector
        stats = importr('stats')
        dff['p_adjust_value'] = stats.p_adjust(FloatVector(dff['p_value']), method='BH')
    except ImportError as e:
        print("R doesn't work")

        # Benjamini and Hochberg（BH）FDR
        # TODO：p_adjust in python
        m =dff['p_value'].count()

        print(m)
        dff['p_rank'] = dff['p_value'].rank(ascending=True)
        dff['p_adjust_value'] = dff['p_value'] * (m / dff['p_rank'])
        dff['p_k'] = 0.05*dff['p_rank'] / m
        min_rank = min(dff[dff['p_adjust_value'] > dff['p_k']]['p_rank'])
        dff[dff['p_rank'] >min_rank]['p_adjust_value'] = dff['p_value']
        # dff=dff.drop(['p_rank','p_k'],axis=1)

    if adjust:
        dff['P'] =dff['p_adjust_value']
        y_text='-log10 ( adjust p )'

    else:
        dff['P']=dff['p_value']
        y_text='-log10 ( p_value )'

    dff.loc[(dff['fd'] > fc) & (dff['P'] < p_value), 'sig'] = 'up'
    dff.loc[(dff['fd'] < -1 * fc) & (dff['P'] < p_value), 'sig'] = 'down'
    # dff.to_csv('../test/all.csv')

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    ax.tick_params(labelsize=13)
    ax = sns.scatterplot(x='fd', y=-np.log10(dff['P']),
                         hue='sig',
                         markers='O',
                         hue_order=('down', 'normal', 'up'),
                         palette=("#377EB8", "grey", "#E41A1C"),
                         data=dff,
                         legend=False)
    ax.set_ylabel(y_text, fontweight='bold',fontsize=13)
    ax.set_xlabel('log2 (fold change)', fontweight='bold',fontsize=13)
    plt.axhline(y=-1*np.log10(p_value),
                color='black',
                linestyle='--',
                linewidth=0.8,
                alpha=0.4
                )
    plt.axvline(x=-1*fc,
                color='black',
                linestyle='--',
                linewidth=0.8,
                alpha=0.4
               )
    plt.axvline(x=fc,
                color='black',
                linestyle='--',
                linewidth=0.8,
                alpha=0.4
                )
    plt.title(str1+'_'+str2,fontsize=14)
    plt.savefig('../test/volcano.pdf',dpi=200)
    dff[dff['sig'] == 'up'].drop('sig',axis=1).to_csv('../test/up.csv')
    dff[dff['sig'] == 'down'].drop('sig',axis=1).to_csv('../test/down.csv')
    plt.show()


def umap():
    # TODO:umap plot
    pass

def pca():
    # TODO:pca plot
    pass



if __name__ == '__main__':
    ## heatmap
    # from sklearn import datasets
    # iris = datasets.load_iris()
    # x, y = iris.data, iris.target
    # pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150, 1))),
    #                        columns=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)',
    #                                 'class'])
    # pd_iris.columns=['E1','F2','G3','H4','J5']
    # heatmapC(pd_iris,'iris_heatmapC')

    ## volcano
    data = pd.read_csv('../test/xx_4758prot_21sample.csv', index_col=0, header=0)
    data[data == 0] = np.nan
    df = np.log2(data)
    group1 = [0, 1, 2, 3, 4, 5, 6]
    group2 = [7, 8, 9, 10, 11, 12, 13]
    volcano(df,group1,group2,adjust=True)

