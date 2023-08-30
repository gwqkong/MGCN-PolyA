
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import numpy as np
from keras.models import  load_model
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import math
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Input, concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

import argparse
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

motif='AATAAA'
seqLength=606
path4='../one hot/tritest38_differ_hg19AATAAA_onehot.npy'
onehot=np.load(path4,allow_pickle=True)
print(onehot.shape)
onehot = np.reshape(onehot,(onehot.shape[0],606,4))


#
# num = int(onehot.shape[0] / 2)
# # negy = np.zeros(15540)
# # posy = np.ones(17167)
# negy = np.zeros(num)
# posy = np.ones(num)
# y_data = np.concatenate((negy, posy), axis=0)
# onehot=onehot.astype(float)
y_data=np.loadtxt('../split data/remove duplicate/hg38_differ_hg19_label.txt')
model=load_model('D:\code\highgate PAS/final_model\model\highway_hs/baseline\deepgengrepAATAAA0.8705014749262537.h5')
pred=model.predict(onehot)
Twoclassfy_evalu1(y_data,pred)

print(onehot.shape)