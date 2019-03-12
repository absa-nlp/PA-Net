# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:19:04 2018

@author: NLP
"""

import pickle


def get_dataset(file_name):
    f = open(file_name,'r',encoding='utf-8')
    data = f.readlines()
    context,label=[],[]
    maxlen = 0
    for j,t in enumerate(data):
        word = t.split()
        label.append(int(word[0]))
        context.append(word[1:])
        if len(word[1:]) > maxlen:
            maxlen = len(word[1:])
    return context, label , maxlen

def read_vec(path):
    f = open(path,'rb')
    dict_name = pickle.load(f)
    return dict_name

    

           
            