# coding:utf-8
""" This module is for Proteome plotting """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def heatmapC(data,filename,method='complete',metric='euclidean',row_cluster=True,col_cluster=False):
    '''
    :param data:
    :param filename:
    :param method:
    :param metric:
    :param row_cluster:
    :param col_cluster:
    :return: file by pdf
    '''
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



def volcano():
    # TODO:volcano plot
    pass

def umap():
    # TODO:umap plot
    pass

def pca():
    # TODO:pca plot
    pass



if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150, 1))),
                           columns=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)',
                                    'class'])
    pd_iris.columns=['E1','F2','G3','H4','J5']
    heatmapC(pd_iris,'iris_heatmapC')



