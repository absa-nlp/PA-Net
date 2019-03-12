# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import LSTM ,GRU
from keras import callbacks
from keras.layers import Bidirectional,Input,concatenate,Flatten,RepeatVector,Lambda,TimeDistributed,add,multiply
from keras.layers.core import Dense, Dropout,Activation,Masking
from keras import backend as K
from keras.optimizers import Adam,RMSprop
from flip_attention import Flip_Attention
from keras.losses import binary_crossentropy
#from target_representation import target_representation_layer


class FA_BiLSTM(object):
    def __init__(self,
                 patience=2,batch_size=32,save_path = None,time = None,
                 n_epoch=8,cluster_size =None ,binary_dim = None, 
                 hidden_unit = None, learning_rate= 0.001):
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.patience = patience
        self.save_path = save_path
        self.time = time
        self.hidden_unit = hidden_unit
        self.learning_rate = learning_rate
    
    def train_model(self,x_train,y_train,
                    x_dev,y_dev):
        
        input_shape = x_train.shape[1:]
        attention_layer = Flip_Attention(name='attlayer')
        
        word = Input(shape=input_shape)
        print('word:',np.shape(word))
        
        mask_word = Masking(mask_value=0.0)(word)
        hidden = Bidirectional(LSTM(units=self.hidden_unit,return_sequences=True,dropout=0.5))(mask_word)
        
        at_done, fa_w, _, w2, _, _ = attention_layer(hidden)
        print('at_done:',np.shape(at_done))
        print('w2:',np.shape(w2))
        x = Dropout(0.3)(at_done)
        x = Dense(1)(x)
        y = Activation('sigmoid')(x)
        model = Model(inputs=word, outputs=y)
        

        RMS = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)
        
        model.compile(loss=customLoss(w2,lamda1=1.0),
                  optimizer=RMS, metrics=['accuracy'])
        saveBestModel = callbacks.ModelCheckpoint(self.save_path+'/model'+self.time+'.h5', monitor='val_loss', verbose=1, mode='auto',save_best_only=True)

        model.fit(x_train,y_train, batch_size=self.batch_size, epochs=self.n_epoch,verbose=1,
                  validation_data=(x_dev,y_dev),callbacks=[saveBestModel])
        
def customLoss(w2, lamda1=0.2):
    def lossFunction(y_true,y_pred):  
        loss1 = binary_crossentropy(y_true, y_pred)
        loss2 = K.mean(K.mean(w2))
        loss = loss1 + lamda1 * loss2
        return loss
    return lossFunction     
        