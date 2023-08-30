import os
import numpy as np
import keras

import h5py
import math
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential, Model
from keras.layers import  Embedding
from keras.layers import Convolution1D, Softmax, Add, Lambda,MaxPooling1D, Dense,Lambda, Dropout,Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from keras import regularizers,optimizers
from nltk import bigrams,trigrams
from os.path import join, abspath, dirname, exists
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Bio import SeqIO
from gensim.models import Word2Vec
from bayes_opt import BayesianOptimization
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import *
import tensorflow as tf
from keras_pos_embd import PositionEmbedding
import re
from sklearn.model_selection import train_test_split
import scipy.io as scio
from keras.layers import GRU
import  xgboost as xgb
import  keras
from keras.callbacks import  ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import backend as K

def Twoclassfy_evalu1(y_test, y_predict1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    aucs = []
    for i in range(len(y_test)):
        if y_predict1[i]> 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict1[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict1[i]< 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 0:
            TN += 1

    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    Acc = (TP + TN) / (TP + FP + TN + FN)

    if TP==0 or TN==0 or FN==0 or FP==0:
        print('TP==0 or TN==0 or FN==0 or FP==0')
        return Sn,Sp,Acc
    else:
        Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
        Precision = TP / (TP + FP)
        F1_score = (2 * Precision * Sn) / (Precision + Sn)
        fpr,tpr,thresholds = roc_curve(y_test,y_predict1)
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        # plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
        # i +=1
        # plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)


        print('TP',TP)
        print('FP',FP)
        print('FN', FN)
        print('TN', TN)

        print('Sn', Sn)
        print('Sp', Sp)
        print('ACC', Acc)
        print('Mcc', Mcc)
        print('Precision',  Precision)
        print('F1_score', F1_score)
        print('AUC', aucs)

        # result = [TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs ]
        # np.savetxt('F:/yuanyuan/AAA/iLearn-master/sequencefeature/CNN_Test_result.txt', result, delimiter=" ", fmt='%s')
        return  TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs
def acc(y_test,y_predict1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    aucs = []
    for i in range(len(y_test)):
        if y_predict1[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict1[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 0:
            TN += 1

    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    return Acc

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>0.0,x,alpha*tf.nn.elu(x))
def Gated_conv1d(seq,kernel_size):
    dim = K.int_shape(seq)[-1]
    h = Convolution1D(dim*2, kernel_size, padding='same')(seq)
    def _gate(seq,h):
        s, h = seq,h
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.sigmoid(g)
        h = selu(h)
        return (1-g) * s + g * h
        # return   s + g * h
    # seq = Lambda(_gate)([seq, h])
    seq = _gate(seq,h)
    return seq



motif='hg38_hs_AATAAA'
i=50
#W2v特征：
texts = []
for index, record in enumerate(SeqIO.parse('..\split data/train/total/'+ motif +'.fa', 'fasta')):
  tri_tokens = trigrams(record.seq)
  temp_str = ""
  for item in ((tri_tokens)):
    # print(item),
    temp_str = temp_str + " " + item[0] + item[1] + item[2]
    # temp_str = temp_str + " " +item[0]
  texts.append(temp_str)
seq = []
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
  doc = re.sub(stop, '', doc)
  seq.append(doc.split())
from gensim import models

# w2v_model = models.KeyedVectors.load_word2vec_format('D:\data\Kalkataw data model\model\Kalkataw hs\H AATAAA_word2vec_'+str(i)+'.model')
w2v_model = Word2Vec.load('../model/Kalkataw hs/hg38_H_3mer ' + motif + '_word2vec_'+str(i)+'.model')
# embedding_matrix = w2v_model.vectors
# vocab_list = list(w2v_model.key_to_index.keys())
embedding_matrix = w2v_model.wv.vectors
vocab_list = list(w2v_model.wv.key_to_index.keys())
word_index = {word: index for index, word in enumerate(vocab_list)}

def get_index(sentence):
  global word_index
  sequence = []
  for word in sentence:
    try:
      sequence.append(word_index[word])
    except KeyError:
      pass
  return sequence


X_data = np.array(list(map(get_index, seq)), dtype=object)  # map是python内置函数，会根据提供的函数对指定的序列做映射。

X_data=X_data.astype(float)
# maxlen=604
# MODEL_PATH = '../'
# # filepath = os.path.join(MODEL_PATH, '../parallel(w2v+sig+pwm+kmer) .h5')
# if not os.path.exists(MODEL_PATH):
#     os.makedirs(MODEL_PATH)

maxlen=604
promodel = Sequential()
promodel.add(Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    input_length=maxlen,
    weights=[embedding_matrix],
    trainable=True))
# L = int((X_data.shape[0]) / 2)
L=int((X_data.shape[0]))
X_data1=promodel.predict(X_data)
print(X_data1.shape)

# X_data=np.reshape(X_data1,(2*L,604,i))

X_data=np.reshape(X_data1,(L,604,i))

def SharedDeepCNNwithShape(shape1=None,  filters=64, ks=3, filter2=23, size2=6, filter3=28, size3=8,
                               filter4=11, size4=5, lstm1=189, penalty=0.005, TRAINX=None, TRAIBY=None, validX=None,
                               validY=None, lr=None):
    w2v_input = Input(shape=shape1,name='w2v_input')
    ini_featuresa0 = Convolution1D(filters, ks, padding='same', activation='selu',kernel_initializer='lecun_normal'
                                  )(w2v_input)
    ini_featuresa1 = Convolution1D(filters, ks + 6, padding='same', activation='selu',kernel_initializer='lecun_normal'
                            )( w2v_input)
    ini_featuresa2 = Convolution1D(filters, ks +12, padding='same', activation='selu',kernel_initializer='lecun_normal'
                            )(w2v_input)
    ini_featuresa = keras.layers.add([ini_featuresa0, ini_featuresa1,ini_featuresa2])

    g_features1 = Gated_conv1d(ini_featuresa, ks)
    # g_features1 = keras.layers.add([g_features1,ini_featuresa2])
    g_features1 = MaxPooling1D(pool_size=2)(g_features1)
    g_features1 = Dropout(0.5)(g_features1)

    ini_featuresa1_1 = Convolution1D(filters, ks, padding='same', activation='selu',kernel_initializer='lecun_normal'
                                   )(g_features1)
    ini_featuresa1_2 = Convolution1D(filters, ks + 6, padding='same', activation='selu',kernel_initializer='lecun_normal'
                                   )(g_features1)
    ini_featuresa1_3 = Convolution1D(filters, ks + 12, padding='same', activation='selu',kernel_initializer='lecun_normal'
                                   )(g_features1)
    ini_featuresa = keras.layers.add([ini_featuresa1_1, ini_featuresa1_2,ini_featuresa1_3])

    g_features2 = Gated_conv1d(ini_featuresa, ks)
    # g_features2 = keras.layers.add([g_features2, ini_featuresa1_3])
    g_features2 = MaxPooling1D(pool_size=2)(g_features2)
    g_features2 = Dropout(0.5)(g_features2)

    ini_featuresa2_1 = Convolution1D(filters, ks, padding='same', activation='selu',
                                   kernel_initializer='lecun_normal'

                                     )(g_features2)
    ini_featuresa2_2 = Convolution1D(filters, ks + 6, padding='same', activation='selu',
                                     kernel_initializer='lecun_normal'
                                     )(g_features2)
    ini_featuresa2_3 = Convolution1D(filters, ks + 12, padding='same', activation='selu',
                                     kernel_initializer='lecun_normal'
                                     )(g_features2)
    ini_featuresa = keras.layers.add([ini_featuresa2_1, ini_featuresa2_2,ini_featuresa2_3])
    g_features3 = Gated_conv1d(ini_featuresa, ks)
    g_features3 = MaxPooling1D(pool_size=2)(g_features3)
    g_features3 = Dropout(0.5)(g_features3)


    out_xin= Flatten()(g_features3)
    #
    Y = Dense(190, activation='relu', name='before_dense')(out_xin)
    Y = Dropout(0.5)(Y)
    output = Dense(2, activation='softmax',name='w2v_output')(Y)

    model = Model(inputs=[w2v_input], outputs=output)
    print(model.summary())
    # model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    # checkpointer = ModelCheckpoint(monitor='val_accuracy', verbose=1, save_best_only=True)
    # model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), metrics=['accuracy'])
    model.compile(loss={'w2v_output': 'categorical_crossentropy'}, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy'])

    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=6, verbose=1)

    earlystopper = EarlyStopping(monitor='val_accuracy', patience=23, verbose=1)
    model.fit([w_train], y_train1, epochs=200, batch_size=200,
              shuffle=True,
              callbacks=[earlystopper, lr_reduce],
              validation_data=({'w2v_input': w_val}, {'w2v_output': y_val1}), verbose=1)
    #
    return model


w_num = X_data.shape[0]
w_len = X_data.shape[1]
w_dim = X_data.shape[2]
input_shape1 = (w_len, w_dim)
# label y
num = int(w_num / 2)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
y_data=to_categorical(y_data)

w_train, w_val, y_train1, y_val1=train_test_split(X_data, y_data, test_size=0.2, random_state=7)
model1 = SharedDeepCNNwithShape(shape1=input_shape1     )
# KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
pred=model1.predict(w_val)
print(pred.shape)
accur=acc(y_val1[:,1],pred[:,1])


model1.save('../final_model/model/highway_dm/' + motif + 'w2v50'+str(accur)+'.h5')
print('model save')









