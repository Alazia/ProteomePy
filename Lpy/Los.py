# coding:utf-8
"""This module is for file reading,saving and renaming"""

import os
import sys
import shutil
import fnmatch

import pandas as pd


def filename_replace(path, filename, old_str, new_str):
    '''
    :param path: where files exist
    :param filename: string contained in the filename, NO re
    :param old_str: old string that you want to replace
    :param new_str: new string that you want to use
    :return: none
    '''

    old_names = os.listdir(path)
    for old_name in old_names:
        if old_name != sys.argv[0]:
            if old_name.__contains__(filename):
                new_name = old_name.replace(old_str, new_str)
                os.rename(os.path.join(path, old_name), os.path.join(path, new_name))
                print(old_name, "has been renamed successfully! New name is: ", new_name)
    return print('Renaming Finished')


def formatSize(bytes):
    '''
    :param bytes: bytes
    :return: size by mb
    '''
    bytes = float(bytes)
    kb = bytes / 1024
    mb = kb / 1024
    return mb


def search_text(search,path):
    '''
    :param search: search list
    :param path: where you want to search
    :return: search result by dict{filename:size by mb}
    '''
    res = []
    size = []
    for root, dirnames, filenames in os.walk(os.path.expanduser(path)):
        for extension in search:
            for filename in fnmatch.filter(filenames, extension):
                res.append(os.path.join(root,filename))
                size.append(formatSize(os.path.getsize(os.path.join(root,filename))))
    res_dict=dict(zip(res,size))
    return res_dict


def search_file(path, w_search=[], s_search=[]):
    '''
    :param path: where you want to search
    :param w_search: search list that you want
    :param s_search: search list that you don't want
    :return:
    '''
    w=search_text(w_search,path)
    s=search_text(s_search,path)
    new=dict(w.items()-s.items())
    df = pd.DataFrame(pd.Series(new), columns=['size'])
    df = df.reset_index().rename(columns={'index': 'filename'})
    return df


def file_move(search, path, target_path):
    count = 0
    for root, dirnames, filenames in os.walk(os.path.expanduser(path)):
        for extension in search:
            for filename in fnmatch.filter(filenames, extension):
                try:
                    shutil.move(path+'\\'+filename,target_path)
                    count += 1
                    print(filename+' has been moved')
                except shutil.Error as e:
                    print(filename+' already exists')
                finally:
                    print('number of files moved: ',count)


def file_copy(df,target_path):
    """
    :param df:data with 'path' and 'filename'
    :param target_path:
    :return:
    """
    print('Total File numbers:',len(df))
    count=0
    try:
        for i in range(0, len(df)):
            shutil.copy(df.iloc[i]['path'] + '\\' + df.iloc[i]['filename'], target_path)
            count += 1
            print(df.iloc[i]['filename'] + ' has been copied\n', count)
    except shutil.Error as e:
        print('file already exists')
    finally:
        print('number of files: ', count)


if __name__ == '__main__':
    path = input('please input path:')
    flag = '1'
    while flag == '1':
        mode = int(input('>>choose mode:\n1:find files\n2:rename files\n3:move files\n>>Your choose:'))
        if mode == 1:
            w_str = input('[Re]search list that you want:\n')
            w = w_str.split(",")
            s_str = input('[Re]search list that you don;t want:\n')
            s = s_str.split(",")
            df = search_file(path, w, s)
            print(df)
            df.to_csv('Found_files.csv', index=0)
        elif mode == 2:
            filename = input('[No Re]filename that you want to rename:\n')
            w_str='*'+filename+'*'
            w = w_str.split(",")
            s=[]
            df=search_file(path,w,s)
            print('The files list:\n', df)
            old_str = input('Old string:\n')
            new_str = input('New string:\n')
            filename_replace(path, filename, old_str, new_str)
        elif mode == 3:
            search = input('[Re]filename that you want to move:\n')
            search = search.split(",")
            target_path = input('Target path:\n')
            file_move(search, path, target_path)
            # TODO:if target_path doesn't exist,filename would be the target_path,to fix it.
        elif mode == 4:
            data = pd.read_csv(input('list_file'), header=0)
            data = data.dropna(axis=0, how='any')
            target_path=(input('target_path'))
            file_copy(data,target_path)
        else:
            print('wrong')
        flag=input('Input 1 to continue,or finished:\n')
        if flag=='1':
            print('=====continue=====')
        else:
            print('Finished')