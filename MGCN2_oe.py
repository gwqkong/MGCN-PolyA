import os
import numpy as np
import h5py
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Embedding, Convolution1D, Dropout, Activation, MaxPooling1D
from keras.layers import Convolution1D, Softmax, Add, Lambda,MaxPooling1D, Dense,Lambda, Dropout,Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from keras import regularizers,optimizers
from nltk import bigrams,trigrams
from os.path import join, abspath, dirname, exists
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Bio import SeqIO
from gensim.models import Word2Vec
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_curve, precision_recall_curve, auc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import *
import tensorflow as tf
from keras_pos_embd import PositionEmbedding
import re
from sklearn.model_selection import train_test_split
import scipy.io as scio
from keras.layers import GRU
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
import keras
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

#第一段




def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>0.0,x,alpha*tf.nn.elu(x))
def Gated_conv1d(seq,kernel_size):
    dim = K.int_shape(seq)[-1]
    h = Convolution1D(dim*2, kernel_size, padding='same')(seq)
    def _gate(seq, h):
        s, h = seq, h
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.sigmoid(g)
        h = selu(h)
        return (1-g) * s + g * h
    # seq = Lambda(_gate)([seq, h])
    seq = _gate(seq, h)
    return seq



def SharedDeepCNNwithShape(shape1=None,  filters=64, ks=3, filter2=23, size2=6, filter3=28, size3=8,
                               filter4=11, size4=5, lstm1=189, penalty=0.005, TRAINX=None, TRAIBY=None, validX=None,
                               validY=None, lr=None):
    onehotenac_input = Input(shape=shape1)
    ini_featuresa3_0 = Convolution1D(filters, ks, padding='same', activation='selu',kernel_initializer='lecun_normal'
                                  )(onehotenac_input)
    ini_featuresa3_1 = Convolution1D(filters, ks + 6, padding='same', activation='selu',kernel_initializer='lecun_normal'
                            )( onehotenac_input)
    ini_featuresa3_2 = Convolution1D(filters, ks + 12, padding='same', activation='selu',kernel_initializer='lecun_normal'
                            )(onehotenac_input)
    ini_featuresa = keras.layers.add([ini_featuresa3_0, ini_featuresa3_1, ini_featuresa3_2])
    g_features1 = Gated_conv1d( ini_featuresa, ks)
    g_features1 = MaxPooling1D(pool_size=2)(g_features1)
    g_features1 = Dropout(0.5)(g_features1)

    ini_featuresa4_1 = Convolution1D(filters, ks, padding='same', activation='selu',kernel_initializer='lecun_normal'

                                   )(g_features1)
    ini_featuresa4_2 = Convolution1D(filters, ks + 6, padding='same', activation='selu',kernel_initializer='lecun_normal'
                                   )(g_features1)
    ini_featuresa4_3 = Convolution1D(filters, ks + 12, padding='same', activation='selu',kernel_initializer='lecun_normal'
                                   )(g_features1)
    ini_featuresa = keras.layers.add([ini_featuresa4_1, ini_featuresa4_2, ini_featuresa4_3])
    g_features2 = Gated_conv1d(ini_featuresa, ks)
    # g_features2 = keras.layers.add([g_features1,g_features2])
    g_features2 = MaxPooling1D(pool_size=2)(g_features2)
    g_features2 = Dropout(0.5)(g_features2)
    g_features2 = Convolution1D(2*filters,1,padding='same')(g_features2)
    g_features2 = MaxPooling1D(pool_size=2)(g_features2)

    ini_featuresa5_1 = Convolution1D(filters, ks, padding='same', activation='selu',
                                    kernel_initializer='lecun_normal'

                                     )(g_features2)
    ini_featuresa5_2 = Convolution1D(filters, ks + 6, padding='same', activation='selu',
                                     kernel_initializer='lecun_normal'
                                     )(g_features2)
    ini_featuresa5_3 = Convolution1D(filters, ks + 12, padding='same', activation='selu',
                                     kernel_initializer='lecun_normal'
                                     )(g_features2)
    ini_featuresa = keras.layers.add([ini_featuresa5_1, ini_featuresa5_2, ini_featuresa5_3])
    g_features3 = Gated_conv1d(ini_featuresa, ks)
    g_features3 = MaxPooling1D(pool_size=2)(g_features3)
    g_features3 = Dropout(0.5)(g_features3)



    out_xin= Flatten()(g_features3)

    Y = Dense(190, activation='relu', name='before_dense')(out_xin)
    Y = Dropout(0.5)(Y)
    output = Dense(1, activation='sigmoid')(Y)

    model = Model(inputs=[onehotenac_input], outputs=output)
    print(model.summary())
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003), metrics=['accuracy'])
    # checkpointer = ModelCheckpoint(monitor='val_accuracy', verbose=1, save_best_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=6, verbose=1)

    earlystopper = EarlyStopping(monitor='val_accuracy', patience=18, verbose=1)
    model.fit([x_train], y_train, epochs=200, batch_size=300,
              shuffle=True,
              callbacks=[earlystopper, lr_reduce],
              validation_data=([x_val], y_val), verbose=1)
    #
    return model


motif='dm_all_polyA'

#ENAC
path='..\ENAC/all/train' + motif + '-ENAC.npy'
enac=np.load(path, allow_pickle=True)
enac=np.reshape(enac, ([enac.shape[0], 602, 4]))
num=int(enac.shape[0]/2)
negy=np.zeros(num)
posy=np.ones(num)
y_data=np.concatenate((negy,posy),axis=0)
# input_shape2=(enac.shape[1],enac.shape[2])
enac = enac.astype(float)
#pwm/BPB+onehot

# path3='../BPB/train/HAATAAA.npy'
# BPB=np.load(path3)
# BPB1=np.reshape(BPB, (BPB.shape[0],2, 606))
# BPB2=BPB1.transpose((0,2,1))
path4='../one hot/all/train' + motif + '-onehot.npy'
onehot=np.load(path4,allow_pickle=True)
onehot = np.reshape(onehot,(onehot.shape[0],606,4))
# BPB_onehot=np.concatenate((BPB2,onehot),axis=-1)
input_shape3=(onehot.shape[1], onehot.shape[2])
# BPB_onehot=onehot.astype(float)
one_enac=np.concatenate((onehot,enac),axis=1)
one_enac=one_enac.astype(float)
input_shape2 =(one_enac.shape[1],one_enac.shape[2])
# x_train, x_val, y_train2, y_val2=train_test_split(one_enac, y_data, test_size=0.2, random_state=7)

#
# def rf_cv(filter1,size1,size2,filter3,size3,densize1):
#     models = SharedDeepCNN(shape2=input_shape2, filter1=int(filter1),size1=int(size1),
#                                     size2=int(size2),filter3=int(filter3),size3=int(size3),densize1=int(densize1)
#                                     ,penalty=0.005)
#     predict = models.predict([x_val])
#     r2 = acc(y_val2,predict)
#     return r2
#
# from hyperopt import hp
# rf_bo = BayesianOptimization(
#     rf_cv,
#     {
#      'filter1':(30,130),
#     'filter3': (30,130),
#      'size1': (3,10),
#      'size2': (3,10),
# 'size3': (3,10),
#     'densize1':(120,300)
#
#      }
# )
# rf_bo.maximize(n_iter=50)
# print(rf_bo.max['target'])
# #查看优化得到的参数
# print(rf_bo.max['params'])
from sklearn.model_selection import StratifiedKFold

# seed=7
# np.random.seed(seed)
# x_train, x_val, y_train, y_val=train_test_split(one_enac, y_data, test_size=0.2, random_state=7)
#
# models = SharedDeepCNNwithShape(shape1=input_shape2,)
# models.save('../final_model/model/highway/5one_enactryyuanshifinalmodel' + str(1) + '.h5')

for l in range(1,3):
    seed = 7
    np.random.seed(seed)
    x_train, x_val, y_train, y_val = train_test_split(one_enac, y_data, test_size=0.2, random_state=7)
    models = SharedDeepCNNwithShape(shape1=input_shape2, )
    # KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred=models.predict(x_val)
    accur=acc(y_val,pred)
    if accur>0.88:
        models.save('../final_model/model/highway_M/'+str(motif)+'one_enactrymodel'+str(accur)+'.h5')
        print('model save')
# KF= StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#
# preds=[]
# # for train_index, val_index in KF.split(one_enac, y_data):
# #     x_train, x_val = one_enac[train_index], one_enac[val_index]
# #     y_train, y_val = y_data[train_index], y_data[val_index]
# #     # s_train,s_val  = BPB_onehot[train_index],BPB_onehot[val_index]
# #
# #
# #     models = SharedDeepCNNwithShape(shape2=input_shape2,shape3=input_shape3)
# #     predict = models.predict([x_val])
# #     preds.append(predict)
# j=1
# for train_index, val_index in KF.split(one_enac, y_data):
#     x_train, x_val = one_enac[train_index], one_enac[val_index]
#     y_train, y_val = y_data[train_index], y_data[val_index]
#     # s_train,s_val  = BPB_onehot[train_index],BPB_onehot[val_index]
#
#
#     models = SharedDeepCNNwithShape(shape1=input_shape2,)
#     models.save('../final_model/model/highway/one_enactryyuanshifinalmodel' + str(j) + '.h5')
#     print(' models.save(../final_model/model/highway/one_enactryyuanshifinalmodel')
#     j = j + 1
#     predict = models.predict([x_val])
#     preds.append(predict)
# SN = []
# SP = []
# ACC = []
# MCC = []
# Precisions = []
# F1_scores = []
# AUC = []
# for pred in preds:
#     TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs=Twoclassfy_evalu1(y_val, pred)
#     SN.append(Sn)
#     SP.append(Sp)
#     ACC.append(Acc)
#     MCC.append(Mcc)
#     Precisions.append(Precision)
#     F1_scores.append(F1_score)
#     AUC.append(aucs)
# meanSN = np.mean(SN)
# meanSP = np.mean(SP)
# meanACC = np.mean(ACC)
# meanPrecision = np.mean(Precisions)
# meanF1_score = np.mean(F1_scores)
# meanMCC = np.mean(MCC)
# meanAUC=np.mean(AUC)
# print("meanSN",meanSN)
# print("meanSP",meanSP)
# print("meanACC",meanACC)
# print('meanPrecision',meanPrecision)
# print('meanF1_score',meanF1_score)
# print("meanMCC",meanMCC)
# print("meanAUC",meanAUC)

