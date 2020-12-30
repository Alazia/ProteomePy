# coding:utf-8
""" This module is for Proteome plotting """
import string

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


from numpy import NaN
import numpy as np


def fig_ax(title='title',x_text='xlabel',y_text='ylabel',fig1=5,fig2=5):
    fig = plt.figure(figsize=(fig1, fig2))
    # ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax = plt.subplot(111)
    ax.tick_params(labelsize=13)
    ax.set_ylabel(y_text, fontweight='bold', fontsize=13)
    ax.set_xlabel(x_text, fontweight='bold', fontsize=13)
    if title==False:
        pass
    else:
        ax.set_title(title, fontsize=14)
    return fig, ax


def heatmap(data, filename, method='complete', metric='euclidean', row_cluster=True, col_cluster=False):
    """
    TODO:add row and col bar.
    :param data:
    :param filename:
    :param method:
    :param metric:
    :param row_cluster:
    :param col_cluster:
    :return: file by pdf
    """
    col_c=dict(zip(list(string.ascii_letters[len(data.columns):]), plt.get_cmap('RdYlBu')(np.linspace(0, 1, len(data.columns)))))
    fig,ax=fig_ax()
    ax=sns.clustermap(data=data,
                   method=method,
                   metric=metric,
                   row_cluster=row_cluster,
                   col_cluster=col_cluster,
                   cmap='RdYlBu',
                   col_colors=pd.Series(data.columns.get_level_values(None), index=data.columns).map(col_c),
                   row_colors=None,
                   cbar_pos=(0.03, 0.76, 0.03, 0.18),
                   yticklabels=False,
                   xticklabels=True
                   )
    plt.savefig('../test/' + filename + '.pdf', dpi=200)
    plt.show()


def volcano(df,group1:list,group2:list,fc=1,p_value=0.05,str1="grp1",str2="grp2",pair=False,adjust=True):
    """
    :param df: np.log2 matrix data after replacing zero with np.nan
    :param group1:
    :param group2:
    :param fc:
    :param p_value:
    :param str1:
    :param str2:
    :param pair:
    :param adjust:
    :return:
    """
    # TODO:param,test 'pair',simplify

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
        print("R doesn't work\nplease install rpy2")
        return None

        # Benjamini and Hochberg（BH）FDR
        # TODO：p_adjust in python
        # m =dff['p_value'].count()
        # dff['p_rank'] = dff['p_value'].rank(ascending=True)
        # dff['p_adjust_value'] = dff['p_value'] * (m / dff['p_rank'])
        # dff['p_k'] = 0.05*dff['p_rank'] / m
        # min_rank = min(dff[dff['p_adjust_value'] > dff['p_k']]['p_rank'])
        # dff[dff['p_rank'] >min_rank]['p_adjust_value'] = dff['p_value']
        # # dff=dff.drop(['p_rank','p_k'],axis=1)

    if adjust:
        dff['P'] = dff['p_adjust_value']
        y_text = '-log10 ( adjust p )'

    else:
        dff['P'] = dff['p_value']
        y_text = '-log10 ( p_value )'

    dff.loc[(dff['fd'] > fc) & (dff['P'] < p_value), 'sig'] = 'up'
    dff.loc[(dff['fd'] < -1 * fc) & (dff['P'] < p_value), 'sig'] = 'down'
    # dff.to_csv('../test/all.csv')

    title = str1 + '_' + str2
    x_text = 'log2 (fold change)'
    fig, ax = fig_ax(title, x_text, y_text)
    ax = sns.scatterplot(x='fd', y=-np.log10(dff['P']),
                         hue='sig',
                         markers='O',
                         hue_order=('down', 'normal', 'up'),
                         palette=("#377EB8", "grey", "#E41A1C"),
                         data=dff,
                         legend=False)

    args = {'color': 'black', 'linestyle': '--', 'linewidth': 0.8, 'alpha': 0.4}
    plt.axhline(y=-1*np.log10(p_value), **args)
    plt.axvline(x=-1*fc, **args)
    plt.axvline(x=fc, **args)
    plt.savefig('../test/volcano.pdf', dpi=200)
    dff[dff['sig'] == 'up'].drop('sig', axis=1).to_csv('../test/up.csv')
    dff[dff['sig'] == 'down'].drop('sig', axis=1).to_csv('../test/down.csv')
    plt.show()


def umap():
    # TODO:umap plot
    pass

def pca(data,label:list):
    """
    :param data:
    :param label:
    :return:
    TODO: to adjust scale
    """
    from sklearn.preprocessing import scale
    data=data.T
    data=scale(data)
    # print(data)
    pc_cols=['pc'+str(i) for i in range(1,data.shape[1]+1,1)]
    df=pd.DataFrame(data,columns=pc_cols)
    from sklearn.decomposition import PCA
    pca_n=PCA(n_components=2)
    pca_df=pca_n.fit_transform(df)
    pc1,pc2=pca_n.explained_variance_ratio_
    print(pca_df)
    dff=pd.DataFrame(pca_df,columns=['PC1','PC2'])
    dff['label']=label

    title='PCA:'+str(data.shape[1])+'features'
    x_text='PC1('+str(format(pc1*100,'.2f'))+'%)'
    y_text='PC2('+str(format(pc2*100,'.2f'))+'%)'
    fig,ax=fig_ax(False,x_text,y_text,6,5)
    ax.set_title(loc='left',label=title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax=sns.scatterplot(x='PC1',y='PC2',hue='label',data=dff)
    plt.savefig('../test/PCA.pdf',dpi=200)
    plt.show()

def density(data,title='Density',color="black",shade=False):
    """
    :param data: matrix data with np.nan(not zero)
    :param title:
    :param color:
    :param shade:
    :return:
    """
    def bw_nrd0(x):
        std = np.std(x)
        q = np.percentile(x, 75)-np.percentile(x, 25)
        i = min(std, q / 1.34)
        if not ((i == std) or (i == abs(x[0])) or (i == 1)):
            i = 1
        return 0.9 * i * len(x) ** -0.2
    df = []
    data.applymap(lambda x: df.append(np.log2(x)) if x==x else x)
    # The bandwidth is a measure of how closely you want the density to match the distribution.
    x_text=' N='+str(len(df))+' Bandwidth='+str(format(bw_nrd0(np.array(df)),'.3f'))
    y_text=' Density '
    fig,ax=fig_ax(title,x_text,y_text)
    ax = sns.kdeplot(df, bw_method='silverman',color=color, shade=shade)
    plt.savefig('../test/'+title+'.pdf',dpi=200)
    plt.show()


def pool_corr(data):
    df=data.corr()
    fig,ax=fig_ax()
    mask = np.zeros_like(df)
    for i in range(1, len(mask)):
        for j in range(0, i):
            mask[i][j] = True
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    ax = sns.heatmap(df,annot=True,fmt='.2f',mask=mask)
    plt.show()


def linear():
    # TODO：linear with ci

    pass

def violin(data,rep:list):
    def cv(x):
        return x.std() / x.mean()
    columns = sum(rep, [])
    print(columns)
    df = data.T
    pd.set_option('mode.chained_assignment', None)
    df['rep'] = df.index
    for i in range(len(rep)):
        df['rep'][rep[i]] = 'rep_' + str(i)
    dff = df.iloc[columns, :]
    dff.index = dff['rep'].values
    dff = dff.drop('rep', axis=1)

    dff = dff.groupby(dff.index).agg(cv).T
    dff.to_csv('../test/cv_rep.csv')
    fix,ax=fig_ax(y_text='cv',fig1=8,fig2=5)
    ax = sns.violinplot(data=dff)
    plt.savefig('../test/cv.pdf', dpi=200)
    plt.show()


if __name__ == '__main__':
    # heatmap
    # from sklearn import datasets
    # iris = datasets.load_iris()
    # x, y = iris.data, iris.target
    # pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150, 1))),
    #                        columns=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)',
    #                                 'class'])
    # pd_iris.columns=['E1','F2','G3','H4','J5']
    # heatmap(pd_iris,'iris_heatmapC')
    # data = pd.read_csv('../test/xx_4758prot_21sample.csv', index_col=0, header=0)
    # data[data == 0] = np.nan
    # df = np.log2(data)
    # m = np.nanmin(df.min().values) * 0.8
    # df.fillna(m, inplace=True)
    # heatmap(df, 'heatmaptest')

    ## volcano
    # data = pd.read_csv('../test/xx_4758prot_21sample.csv', index_col=0, header=0)
    # data[data == 0] = np.nan
    # df = np.log2(data)
    # group1 = [0, 1, 2, 3, 4, 5, 6]
    # group2 = [7, 8, 9, 10, 11, 12, 13]
    # volcano(df,group1,group2,adjust=True)
    #
    # # Density plot
    # data = pd.read_csv('../test/xx_4758prot_21sample.csv', index_col=0, header=0)
    # data[data == 0] = np.nan
    # density(data)

    ## pca
    # data = pd.read_csv('../test/xx_4758prot_21sample.csv', index_col=0, header=0)
    # label=['C','C','C','C','C','C','C','C','C','C','C','C','C','C','N','N','N','N','N','N','N']
    # pca(data,label)
    # pool_corr(data)

    ## umap

    data = pd.read_csv('../test/xx_4758prot_21sample.csv', header=0, index_col=0)
    print(data)
    data[data==0]=np.nan
    rep = [[1, 2, 3], [4, 5, 6],[7,8],[9,10]]
    violin(data,rep)