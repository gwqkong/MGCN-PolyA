import numpy as np
import random as rn
import numpy
from keras.layers import *
import keras
import warnings
# from keras import initializers
from keras.models import Model
from tensorflow.keras import layers, initializers
from keras.layers import *
from keras.layers.noise import *
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras import backend as K
import numpy as np
from Bio import SeqIO
from nltk import trigrams, bigrams,fivegrams
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import  load_model
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import math

class MultiHeadSelfAttention(Layer):
    def __init__(self, n_head, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_head = n_head
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # 初始化权重矩阵
        self.query_dense = Dense(units=d_model, activation=None)
        self.key_dense = Dense(units=d_model, activation=None)
        self.value_dense = Dense(units=d_model, activation=None)
        self.dropout = Dropout(rate=dropout_rate)
        self.output_dense = Dense(units=d_model, activation=None)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_head': self.n_head,
            'd_model': self.d_model
        })
        return config
    def call(self, inputs):
        # 获取输入
        q, k, v = inputs

        # 获取batch size和sequence length
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        # 通过全连接层获取query、key、value向量
        q = self.query_dense(q)  # (batch_size, seq_len, d_model)
        k = self.key_dense(k)  # (batch_size, seq_len, d_model)
        v = self.value_dense(v)  # (batch_size, seq_len, d_model)

        # 将向量拆分成多个头，进行并行计算
        q = tf.reshape(q, shape=(batch_size, seq_len, self.n_head, self.d_model // self.n_head))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, shape=(batch_size, seq_len, self.n_head, self.d_model // self.n_head))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.reshape(v, shape=(batch_size, seq_len, self.n_head, self.d_model // self.n_head))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # 计算scaled dot-product attention
        scale_factor = tf.math.sqrt(tf.cast(self.d_model / self.n_head, dtype=tf.float32))
        k_t = tf.transpose(k, perm=[0, 1, 3, 2])
        qk = tf.matmul(q, k_t) / scale_factor
        attn = tf.nn.softmax(qk, axis=-1)
        attn = self.dropout(attn)
        output = tf.matmul(attn, v)

        # 将头合并，通过全连接层获得输出向量
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, shape=(batch_size, seq_len, self.d_model))
        output = self.output_dense(output)
        return output
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
i=90
texts = []
for index, record in enumerate(SeqIO.parse('..\split data/remove duplicate/hg38_differ_hg19.fa', 'fasta')):
  tri_tokens = fivegrams(record.seq)
  temp_str = ""
  for item in ((tri_tokens)):
    # print(item),
    temp_str = temp_str + " " + item[0] + item[1] + item[2] + item[3] + item[4]
    # temp_str = temp_str + " " +item[0]
  texts.append(temp_str)
seq = []
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
  doc = re.sub(stop, '', doc)
  seq.append(doc.split())
from gensim import models

# w2v_model = models.KeyedVectors.load_word2vec_format('D:\data\Kalkataw data model\model\Kalkataw hs\H AATAAA_word2vec_'+str(i)+'.model')
w2v_model = Word2Vec.load('../model/Kalkataw hs/PASnet ' + motif + '_word2vec_'+str(i)+'.model')

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

w_num = X_data.shape[0]
w_len = X_data.shape[1]
# w_dim = X_data.shape[2]
input_shape1 = (w_len)
# label y
num = int(w_num / 2)
# negy = np.zeros(15540)
# posy = np.ones(17167)
# negy = np.zeros(num)
# posy = np.ones(num)
# y_data = np.concatenate((negy, posy), axis=0)
from tensorflow.keras.utils import to_categorical
y_data=np.loadtxt('../split data/remove duplicate/hg38_differ_hg19_label.txt')
y_data=to_categorical(y_data)

model=load_model('D:\code\highgate PAS/final_model\model\highway_hs/baseline\PASnetAATAAA0.8589970501474926.h5', custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})

pred=model.predict(X_data)
Twoclassfy_evalu1(y_data[:,1],pred[:,1])

print(X_data.shape)