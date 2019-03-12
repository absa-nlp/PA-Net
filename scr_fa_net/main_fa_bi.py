# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
from keras.models import load_model
from fa_bilstm import FA_BiLSTM,customLoss
import os
import logging
import gc
import random
from reader import get_dataset, read_vec
#from reader_2 import get_dataset, read_vec # suit for sst2 and imdb
from keras import backend as K
from keras.models import Model
from flip_attention import Flip_Attention
from evaluation import accuracy
def word_embeding(comment_cut,maxlen,word_vectors):
   
    word_embed = np.zeros((len(comment_cut),maxlen,300))
    
    for i,sentence in enumerate(comment_cut):
        index = 0
        for n,word in enumerate(sentence[:maxlen]):
            try:
                if index < maxlen:
                    word_embed[i][index][:] = word_vectors[word]
                    index += 1
            except:
                if index < maxlen:
                    word_embed[i][index][:] = word_vectors['UNK']
                    index += 1
                continue
    return word_embed

def shuffle_data(data):
    re_index = [i for i in range(len(data[0]))]
    random.shuffle(re_index)
    data_shuffle = []
    for t in data:
        one_shuffle = [t[re_index[i]]  for i in range(len(t)) ] 
        data_shuffle.append(np.array(one_shuffle))
    return data_shuffle
def load_file(file_name):
    data=pd.read_excel(file_name,index=None)
    comment = list(data['comment'])
    label = np.array(data.loc[:,['label']])
    context=[]
    for j,t in enumerate(comment):
        word = t.split()
        context.append(word)

    return context, label
if __name__=='__main__':
    print('Loading Data...')
    logging.basicConfig(filename='log/test.log', filemode="w", level=logging.DEBUG)
    save_path = 'model_data'
    
    train_path= '../mr_data/MR-train_sentence_score.xlsx'
    dev_path= '../mr_data/MR-dev_sentence_score.xlsx'
    test_path= '../mr_data/MR-test_sentence_score.xlsx'
    vec_path = "../mr_data/mr_vec.pkl"
    path = '../mr_data/mr_bais_test.xlsx'
    
    train_context,y_train,train_maxlen = get_dataset(train_path)
    dev_context,y_dev,dev_maxlen = get_dataset(dev_path)
    test_context,y_test,test_maxlen = get_dataset(test_path)
    word_vectors = read_vec(vec_path)
#    maxlen = 200 # suit for imdb
    maxlen = max(train_maxlen, dev_maxlen, test_maxlen)


    x_train = word_embeding(train_context,maxlen,word_vectors)
    x_dev = word_embeding(dev_context,maxlen,word_vectors)
    x_test = word_embeding(test_context,maxlen,word_vectors)
    
#    train_shuffle = shuffle_data([x_train, y_train])
#    dev_shuffle = shuffle_data([x_dev, y_dev])
    test_data,label = load_file(path)
    test_embed = word_embeding(test_data,maxlen,word_vectors)
    times = 5 # training times
    for time in range(times):
        K.clear_session()

        network = FA_BiLSTM(patience=10,batch_size=32,n_epoch=25,
                          save_path=save_path,time=str(time),
                          hidden_unit = 50,
                          learning_rate = 0.001)
    
        print('Training Data...') 
        network.train_model(x_train,y_train,
                            x_dev,y_dev)

    files= os.listdir(save_path)
    mean_acc = 0
    mean_b1 = 0
    mean_b2 = 0
    mean_b3 = 0
    for i,file in enumerate(files):
        K.clear_session()
        model = load_model(save_path+'/'+file,custom_objects={"Flip_Attention": Flip_Attention,
                                                              "lossFunction":customLoss(K.variable(np.ones((1,1))),0.2)})
        evaluate = model.evaluate(x_test,y_test)
        score = model.predict(test_embed)
        mean_acc += evaluate[1]
        bais_acc1 = accuracy(score, label, 0.45, 0.55)
        mean_b1 += float(bais_acc1)
        bais_acc2 = accuracy(score, label, 0.40, 0.60)
        mean_b2 += float(bais_acc2)
        bais_acc3 = accuracy(score, label, 0.35, 0.65)
        mean_b3 += float(bais_acc3)
        print(file,evaluate[1],bais_acc1,bais_acc2,bais_acc3)
#        if evaluate[1] > max_acc:
#            max_acc = evaluate[1]
#            best_name = file
        del model
        gc.collect()
        logging.info(file+' '+str(evaluate[1])+ ' ' +str(bais_acc1)+str(bais_acc2)+str(bais_acc3))
    print('mean_acc:', mean_acc/times)
    print('mean_b1:', mean_b1/times)
    print('mean_b2:', mean_b2/times)
    print('mean_b3:', mean_b3/times)
    logging.info(str(mean_acc/times)+' '+str(mean_b1/times)+' '+str(mean_b2/times)+' '+str(mean_b3/times))    
   

    