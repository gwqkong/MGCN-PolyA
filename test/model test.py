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
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from utils import *
import tensorflow as tf
from keras_pos_embd import PositionEmbedding
import re
from sklearn.model_selection import train_test_split
import scipy.io as scio
from keras.layers import GRU
import  xgboost as xgb
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
    if TP==0 or FN==0 or TN==0 or FP==0:
        print('TP==0 or FN==0 or TN==0 or FP==0')
        return Sn,Sp,Acc
    else:
        Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
        Precision = TP / (TP + FP)
        F1_score = (2 * Precision * Sn) / (Precision + Sn)
        fpr,tpr,thresholds = roc_curve(y_test,y_predict1)
        roc_auc=auc(fpr,tpr)
        precision, recall, thresholds = precision_recall_curve(y_test,y_predict1)

        auprc = auc(recall, precision)
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
        print('auprc',auprc)

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
    h = Convolution1D(dim*3, kernel_size, padding='same')(seq)
    def _gate(x):
        s, h = x
        g, h ,s= h[:, :, :dim], h[:, :, dim:2*dim],h[:,:,2*dim:3*dim]
        g = K.sigmoid(g)
        h = selu(h)
        return (1-g) * s + g * h
    seq = Lambda(_gate)([seq, h])
    return seq
#第一段

motif='bt_AATAAA'
# path='D:\code\PAS feature\signal\Features-Human-'+motif+'-0.8_polyA.mat'
# data=scio.loadmat(path)
# print(data.keys())
# sigdata=np.array(data.get('features'))
# print(sigdata.shape)
# y_data=sigdata[:,-1]
# sigx_data=sigdata[:,:-1]
# BPB=np.load('D:\论文\数据\PAS data seq\CD-HIT 0.8/'+motif+'-BPB.npy')
# print(BPB.shape)
# singal_bpb=np.concatenate((sigx_data,BPB),axis=1)
#W2v特征：
texts = []
for index, record in enumerate(SeqIO.parse('..\split data/test/total/'+motif+'.fa', 'fasta')):
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
w2v_model = Word2Vec.load('..\model/Kalkataw bt/H_3mer '+motif+'_word2vec_50.model')
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
L = int((X_data.shape[0]) / 2)

X_data1=promodel.predict(X_data)
print(X_data1.shape)
i=50
X_data=np.reshape(X_data1,(2*L,604,i))
w_num = X_data.shape[0]
w_len = X_data.shape[1]
w_dim = X_data.shape[2]
input_shape1 = (w_len, w_dim)
#ENAC
path='../ENAC/test'+ motif+'-ENAC.npy'
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
#onehot
path4='../one hot/test'+ motif+'-onehot.npy'
onehot=np.load(path4,allow_pickle=True)
onehot = np.reshape(onehot,(onehot.shape[0],606,4))
# BPB_onehot=np.concatenate((BPB2,onehot),axis=-1)
# input_shape3=(onehot.shape[1], onehot.shape[2])
# BPB_onehot=onehot.astype(float)
one_enac=np.concatenate((onehot,enac),axis=1)
one_enac=one_enac.astype(float)
input_shape2 =(one_enac.shape[1],one_enac.shape[2])
print(input_shape1,input_shape2)




path1='..\K_mer/test'+ motif+'.npy'
k_mer=np.load(path1)
k_mer = k_mer.astype(float)

input_shape3 = (k_mer.shape[1])
print(input_shape3)
from keras.models import  load_model
from keras_bert import  get_custom_objects
import joblib
# model1=load_model('../final_model/model/highway/w2vtryduliyuanshifinalmodel021.h5',custom_objects=get_custom_objects())

# model1=load_model('../final_model/model/highway_M/w2v50—finalmodel030.8607372732592159.h5',custom_objects=get_custom_objects())
model1=load_model('../final_model/model/highway_bt/bt_AATAAAw2v50—finalmodel030.8382415706359369.h5',custom_objects=get_custom_objects())
# model1=load_model('../final_model/model/highway/w2v50—tryduliyuanshifinalmodel036.h5',custom_objects=get_custom_objects())

# model2 = load_model('../final_model/model/onehot_enac/4one_enactryyuanshifinalmodel1.h5',custom_objects=get_custom_objects())
model2 = load_model('../final_model/model/highway_bt/bt_AATAAAone_enactrymodel0.8190354246692275.h5',custom_objects=get_custom_objects())

# model2 = load_model('../final_model/model/highway_M/one_enactrymodel3.h5',custom_objects=get_custom_objects())

w2v_final_model = Model(inputs=model1.input, outputs=model1.get_layer('before_dense').output)
one_enac_final_model = Model(inputs=model2.input, outputs=model2.get_layer('before_dense').output)

w2v_final_vec = w2v_final_model.predict(X_data)
one_enac_final_vec =one_enac_final_model(one_enac)

print(w2v_final_vec.shape,one_enac_final_vec.shape)

w2v_kmer = concatenate((w2v_final_vec,k_mer))

#
w2v_onehotenac=concatenate((w2v_final_vec,one_enac_final_vec))
#
#
y_val = y_data
#
# onehot_kmer = concatenate((before_one_enac,x_train))
# onehot_kmer_val = concatenate((predict_w2v,x_val))
#所有特征结合：
w2v_onehot_kmer =concatenate((w2v_onehotenac,k_mer))

# w2v_onehot_kmer_val =concatenate((w2v_onehotenac_val,x_val))
# rf_w2v_kmer=joblib.load(r'../final_model\model/rf_w2v_kmer_hAATAAAacc.m')
# pred= rf_w2v_kmer.predict_proba(w2v_kmer)
# pred_n0=pred[:,1]
# Twoclassfy_evalu1(y_val, pred_n0)
# rf_w2v_onehotenac=joblib.load(r'../final_model\model/rf_w2v_onehotenac_hAATAAAacc.m')
# pred1 =rf_w2v_onehotenac.predict_proba(w2v_onehotenac)
# pred_n1=pred1[:,1]
#
# Twoclassfy_evalu1(y_val, pred_n1)

# rf_w2v_onehot=joblib.load(r'../final_model\model/rftry_w2v50_onehot__hAATAAAacc2.m')
# rf_w2v_kmer=joblib.load(r'../final_model\model/rftry_w2v50_kmer__hAATAAAacc2.m')

rf_w2v_onehot_kmer=joblib.load(r'../final_model\model/newrftry_w2v50_onehot_kmer_'+ motif+'acc2.m')
pred2 = rf_w2v_onehot_kmer.predict_proba(w2v_onehot_kmer)
# svm_w2v_onehot_kmer = joblib.load(r'../final_model\model/svm_w2v_onehot_kmer_hAATAAAacc1.m')
# svm_pred=svm_w2v_onehot_kmer.predict_proba(w2v_onehot_kmer)
# # pred_n2=pred2[:,1]
# xgb_w2v_onehot_kmer=joblib.load(r'../final_model\model/xgb_w2v_onehot_kmer_hAATAAAacc1.m')
# xgb_pred=xgb_w2v_onehot_kmer.predict_proba(w2v_onehot_kmer)
# print(pred2[:,1])
# integration= np.array([pred2[:,1],svm_pred[:,1],xgb_pred[:,1]])
# print(integration.shape)
# integration=integration.T
# logist=joblib.load(r'../final_model\model/logistic_w2v_onehot_kmer_hAATAAAacc1.m')
# logist_pred=logist.predict(integration)
#
# # print(pred2[1])
# # print(pred2.shape)
Twoclassfy_evalu1(y_val, pred2[:,1])
# Twoclassfy_evalu1(y_val,logist_pred)
# Twoclassfy_evalu1(y_val,svm_pred[:,1])
# rf_svm=(pred2[:,1]+svm_pred[:,1])/2
# Twoclassfy_evalu1(y_val,rf_svm)
# final_pred = (pred+pred1+pred2)/3
# final_pred = (pred_n0+pred_n1+pred_n2)/3
#
# Twoclassfy_evalu1(y_val, final_pred)

# w2v_test_pred=model1.predict(X_data)
# print(w2v_test_pred.shape)
# Twoclassfy_evalu1(y_val,w2v_test_pred[:,1])