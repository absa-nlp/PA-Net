# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 12:53:58 2018

@author: Administrator
"""
import pandas as pd
import numpy as np
#from sklearn.metrics import accuracy_score
#from baidu_trans import baidu_translate
def load_file(file_name):
    data=pd.read_excel(file_name,index=None)
    true = np.array(data.loc[:,['label']])
    bi = np.array(data.loc[:,['bi']])
    att_bi=np.array(data.loc[:,['att-bi']])
    fa_bi = np.array(data.loc[:,['fa-bi']])
    tcn = np.array(data.loc[:,['tcn']])
    fa_tcn = np.array(data.loc[:,['fa-tcn']])
    dpcnn = np.array(data.loc[:,['dpcnn']])
    fa_dpcnn = np.array(data.loc[:,['fa-dpcnn']])
    trans = np.array(data.loc[:,['trans']])
    fa_trans = np.array(data.loc[:,['fa-trans']])
    return true, bi, att_bi, fa_bi, tcn, fa_tcn, dpcnn, fa_dpcnn, trans, fa_trans

def score_to_label(score, min_v, max_v):
    label = np.zeros_like(score)
    for i in range(len(score)):
        if score[i] > max_v:
            label[i] = 2
        elif score[i] < max_v and score[i] > min_v:
            label[i] = 1
    
    return label
def accuracy_score(true_lab, pred_lab):
    count = 0
    for i in range(len(true_lab)):
        if true_lab[i] == pred_lab[i]:
            count += 1
    acc = float(count / len(true_lab))*100
    acc = ("%.2f" % acc)
    return acc

def accuracy(true, pred, min_v, max_v):
    true_lab = score_to_label(true, min_v, max_v)
    pred_lab = score_to_label(pred, min_v, max_v)
    acc = accuracy_score(true_lab, pred_lab)
    return acc
    
#file_name = 'bais_test.xlsx'
#min_v = 0.35
#max_v = 0.65
#true, bi, att_bi, fa_bi, tcn, fa_tcn, dpcnn, fa_dpcnn, trans, fa_trans = load_file(file_name)
#
#bi_acc = accuracy(true, bi, min_v, max_v)
#att_bi_acc = accuracy(true, att_bi, min_v, max_v)
#fa_bi_acc = accuracy(true, fa_bi, min_v, max_v)
#
#tcn_acc = accuracy(true, tcn, min_v, max_v)
#fa_tcn_acc = accuracy(true, fa_tcn, min_v, max_v)
#
#dpcnn_acc = accuracy(true, dpcnn, min_v, max_v)
#fa_dpcnn_acc = accuracy(true, fa_dpcnn, min_v, max_v)
#
#trans_acc = accuracy(true, trans, min_v, max_v)
#fa_trans_acc = accuracy(true, fa_trans, min_v, max_v)
#
#all_acc = [bi_acc, att_bi_acc, fa_bi_acc, dpcnn_acc, fa_dpcnn_acc, tcn_acc, fa_tcn_acc, trans_acc, fa_trans_acc]
#print('bi_acc:',bi_acc)
#print('att_bi_acc:',att_bi_acc)
#print('fa_bi_acc:',fa_bi_acc)
#print('dpcnn_acc:',dpcnn_acc)
#print('fa_dpcnn_acc:',fa_dpcnn_acc)
#print('tcn_acc:',tcn_acc)      
#print('fa_tcn_acc:',fa_tcn_acc)
#print('trans_acc:',trans_acc)
#print('fa_trans_acc:',fa_trans_acc)