# -*- coding: utf-8 -*-
# @Author  : clq
# @FileName: tools.py
# @Software: PyCharm

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from tcn.tcn import TCN
from layers import MultiHeadAttention
import numpy as np
my_seed = 42
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)

def ourmodel():
    in_put = Input(shape=(97, 150))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    # d = Bidirectional(LSTM(128,return_sequences=True))(c)
    d = TCN(nb_filters=128, kernel_size=5, dropout_rate=0.3, nb_stacks=1,  dilations=[1, 2, 4,8], return_sequences=True,activation='relu',padding='same',use_skip_connections=True)(c)
    d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
                           return_multi_attention=False, name='Multi-Head-Attention')(d)

    d = Flatten()(d)
    e = Dense(128, activation='relu', name='FC3')(d)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])