# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:19:04 2018

@author: NLP
"""

import pickle
import pandas as pd

def get_dataset(file_name):
    data=pd.read_excel(file_name,index=None)
    comment = list(data['comment'])
    label = list(data['label'])
    context=[]
    maxlen = 0
    for j,t in enumerate(comment):
        word = t.split()
#        label.append(int(word[0]))
        context.append(word)
        if len(word) > maxlen:
            maxlen = len(word)
    return context, label , maxlen

def read_vec(path):
    f = open(path,'rb')
    dict_name = pickle.load(f)
    return dict_name

    

           
            