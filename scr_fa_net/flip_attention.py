# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:04:10 2018

@author: Administrator
"""

from keras import backend as K
import numpy as np
from keras.engine.topology import Layer
from keras.layers.core import activations
from keras.models import Model
from keras.layers.core import Dense,Activation
from keras.layers import Input,Flatten,RepeatVector,Permute,subtract,multiply,add,concatenate,Lambda
#import numpy as np


class Flip_Attention(Layer):# sig1(h)p1 +(1-sig1(h))[sig2(h)p2 + (1-sig2(h)p2)p3]
    """
    # Inputs:
        Tensor with shape [(batch_size, time_steps, hidden_size)]
    # Returns:
        Tensor with shape (batch_size, hidden_size)
        If return attention weight,
        an additional tensor with shape (batch_size, time_steps) will be returned.
    """    
    def __init__(self,
                 activation=True,
                 use_bias=True,
                 **kwargs):
        self.activation = activation
        self.use_bias = use_bias
        super(Flip_Attention, self).__init__(**kwargs)
     
    def build(self,input_shape):
        if len(input_shape) < 3:
            raise ValueError(
                "Expected input shape of `(batch_size, time_steps, features)`, found `{}`".format(input_shape))
        # Create a trainable weight variable for this layer.
        self.w1 = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],1),
                                      initializer="glorot_normal",
                                      trainable=True)
        self.w2 = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],1),
                                      initializer="glorot_normal",
                                      trainable=True)
        if self.use_bias:
            self.b1 = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer="zeros",
                                        trainable=True)
            self.b2 = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer="zeros",
                                        trainable=True)
        else:
            self.b1 = None
            self.b2 = None
#        self.shape = input_shape
        super(Flip_Attention, self).build(input_shape)
        
    def call(self, inputs, mask=None):

        sigmoid1 = K.sigmoid(K.bias_add(K.dot(inputs,self.w1),self.b1))  #[(batch_size, time_steps, 1)]
        
        w1_weights = K.squeeze(sigmoid1, axis=-1)
#        print('w1_weights:',np.shape(w1_weights))
        ones = K.ones_like(sigmoid1) #[(batch_size, time_steps, 1)] 
        zeros = K.zeros_like(sigmoid1) #[(batch_size, time_steps, 1)]
        w2_weights = K.squeeze(ones-sigmoid1, axis=-1)  #[(batch_size, time_steps)] 
#        print('w2_weights:',np.shape(w2_weights))
#        print('s1_w:',np.shape(s1_w))
        mid_lab = K.concatenate([zeros,ones,zeros], -1)
        pos_lab = K.concatenate([ones,zeros,zeros], -1)
        neg_lab = K.concatenate([zeros,zeros,ones], -1)
        
        sigmoid1 = K.repeat_elements(sigmoid1, 3, -1)  #[(batch_size, time_steps, 3)]
#        print('s1:',np.shape(s1))
        one_constant = K.ones_like(sigmoid1)
#        print(2)
#        print('s1_0:',np.shape(s1_0))
        sigmoid1_sub = one_constant-sigmoid1  # 1- sigmoid1 [(batch_size, time_steps, 3)]
#        s1_1 = subtract([s1_0, s1])   # 1-sig2
        
        sigmoid2 = K.sigmoid(K.bias_add(K.dot(inputs,self.w2),self.b2))
        w3_weights = K.squeeze(sigmoid2, axis=-1)
#        print('w3weights:',np.shape(w3_weights))
        w4_weights = K.squeeze(ones-sigmoid2, axis=-1)
#        print('w4_weights:',np.shape(w4_weights))
        sigmoid2 = K.repeat_elements(sigmoid2, 3, -1) # sig2 [(batch_size, time_steps, 3)
        
        sigmoid2_sub = one_constant-sigmoid2  # 1- sigmoid2 [(batch_size, time_steps, 3)]
        
        mut1_0 = multiply([sigmoid1, mid_lab])  #[(batch_size, time_steps, 3)]   sigmoid1*P1
        mut2_1 = multiply([sigmoid2, pos_lab])  #[(batch_size, time_steps, 3)]   sigmoid2*P2
        mut2_2 = multiply([sigmoid2_sub, neg_lab])  #[(batch_size, time_steps, 3)]   (1-sigmoid2)*P3
        add2 = add([mut2_1,mut2_2])        #[(batch_size, time_steps, 3)]   sigmoid2*P2 + (1-sigmoid2)*P3
        
        mut1_1 = multiply([sigmoid1_sub,add2]) #[(batch_size, time_steps, 3)] (1-sigmoid1)[sigmoid2*P2 + (1-sigmoid2)*P3]
        add_all = add([mut1_0,mut1_1])       #[(batch_size, time_steps, 3)] sigmoid1*P1+(1-sigmoid1)[sigmoid2*P2 + (1-sigmoid2)*P3]
        
#        flip_att = K.sqrt(K.sum(K.square(add_all - mid_lab), axis=-1))  # distance
        flip_att = K.sqrt(K.maximum(K.sum(K.square(add_all - mid_lab), axis=-1), K.epsilon()))  # distance
#        print(np.shape(flip_att))
        if mask is not None:
            flip_att *= K.cast(mask, K.floatx())
        
#        flip_att = flip_att / (K.sum(flip_att, axis=1, keepdims=True) + K.epsilon())
        atx = inputs * K.expand_dims(flip_att, axis=-1)
        output = K.sum(atx, axis=1)
#        print(inputs)
        return  [output, flip_att,w1_weights, w2_weights, w3_weights, w4_weights]

    def compute_mask(self, input, input_mask=None):
        return None
        
    def compute_output_shape(self,input_shape):
        output_len = input_shape[2]
        return [(input_shape[0], output_len), 
                (input_shape[0], input_shape[1]), 
                (input_shape[0], input_shape[1]), 
                (input_shape[0], input_shape[1]), 
                (input_shape[0], input_shape[1]), 
                (input_shape[0], input_shape[1])]