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

seed = 2019
L2_value=0.0001

motif='AATAAA'
i=90
texts = []
for index, record in enumerate(SeqIO.parse('..\split data/train/total/h'+ motif +'.fa', 'fasta')):
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
print(X_data.shape)
# maxlen=604
# MODEL_PATH = '../'
# # filepath = os.path.join(MODEL_PATH, '../parallel(w2v+sig+pwm+kmer) .h5')
# if not os.path.exists(MODEL_PATH):
#     os.makedirs(MODEL_PATH)

maxlen=602
# promodel = Sequential()
# promodel.add(Embedding(
#     input_dim=embedding_matrix.shape[0],
#     output_dim=embedding_matrix.shape[1],
#     input_length=maxlen,
#     weights=[embedding_matrix],
#     trainable=False))
# L=int((X_data.shape[0]))
# X_data1=promodel.predict(X_data)
# print(X_data1.shape)

# X_data=np.reshape(X_data1,(2*L,604,i))

# X_data=np.reshape(X_data1,(L,604,i))
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
def build_PASNet( shape1,embedding_matrix,dropout_rate=0.1,filters=32,l_r= 0.000003,ks=45,units=64):
    # build the PASNet framework
    print('<<<<<<<<<<<<<<<embedding>>>>>>>>>>>>>')
    embedding_layer = Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    input_length=maxlen,
    weights=[embedding_matrix],
    trainable=False)
    input_sequences = Input(shape=shape1, dtype='float32')
    embed_features = embedding_layer(input_sequences)
    embed_features = Dropout(dropout_rate)(embed_features)
    #1
    ini_featuresa0 = Conv1D(filters, ks, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(embed_features)
    ini_featuresa1 = Conv1D(filters, ks+6, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(embed_features)
    ini_featuresa2 = Conv1D(filters, ks+12, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(embed_features)
    ini_featuresa = keras.layers.add([ini_featuresa0,ini_featuresa1,ini_featuresa2]) #求和
    g_features1 = Gated_conv1d(ini_featuresa, ks)
    g_features1 = Dropout(dropout_rate)(g_features1)
    #2
    ini_featuresb0 = Conv1D(filters, ks, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features1)
    ini_featuresb1 = Conv1D(filters, ks+6, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features1)
    ini_featuresb2 = Conv1D(filters, ks+12, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features1)
    ini_featuresb = keras.layers.add([ini_featuresb0,ini_featuresb1,ini_featuresb2])
    g_features2 = Gated_conv1d(ini_featuresb, ks)
    g_features2 = Dropout(dropout_rate)(g_features2)
    #3
    ini_featuresc0 = Conv1D(filters, ks, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features2)
    ini_featuresc1 = Conv1D(filters, ks+6, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features2)
    ini_featuresc2 = Conv1D(filters, ks+12, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features2)
    ini_featuresc = keras.layers.add([ini_featuresc0,ini_featuresc1,ini_featuresc2])
    g_features3 = Gated_conv1d(ini_featuresc, ks)
    # g_features3 = Dropout(dropout_rate)(g_features3)
    # #4
    # ini_featuresd0 = Conv1D(filters, ks, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features3)
    # ini_featuresd1 = Conv1D(filters, ks+6, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features3)
    # ini_featuresd2 = Conv1D(filters, ks+12, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features3)
    # ini_featuresd = keras.layers.add([ini_featuresd0,ini_featuresd1,ini_featuresd2])
    # g_features4 = Gated_conv1d(ini_featuresd, ks)
    # #5
    # ini_featurese0 = Conv1D(filters, ks, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features4)
    # ini_featurese1 = Conv1D(filters, ks+6, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features4)
    # ini_featurese2 = Conv1D(filters, ks+12, padding='same', activation = 'selu', kernel_initializer = initializers.lecun_normal(seed = seed), kernel_regularizer=l2(L2_value))(g_features4)
    # ini_featurese = keras.layers.add([ini_featurese0,ini_featurese1,ini_featurese2])
    # g_features5 = Gated_conv1d(ini_featurese, ks)

    # s_features = Attention(8,16)([embed_features, embed_features, embed_features])
    s_features = MultiHeadSelfAttention(8,16)([embed_features, embed_features, embed_features])

    # s_features = Attention(8,16)([g_features2, g_features2, g_features2]
    features = keras.layers.concatenate([s_features, g_features3])
    # features = Attention(8,16)([g_features3, g_features3, g_features3])

    all_features = Flatten()(features)
    all_features = Dropout(dropout_rate)(all_features)
    fc1 = Dense(units, activation='selu',kernel_initializer = initializers.lecun_normal(seed = seed))(all_features)
    output = Dense(2,activation='softmax')(fc1)
    PASNet = Model(inputs=[input_sequences], outputs=[output])
    optimizer =  tf.keras.optimizers.Adam(lr=l_r, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    PASNet.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    PASNet.summary()
    PASNet.get_config()
    return PASNet


w_num = X_data.shape[0]
w_len = X_data.shape[1]
# w_dim = X_data.shape[2]
input_shape1 = (w_len)
# label y
num = int(w_num / 2)
# negy = np.zeros(15540)
# posy = np.ones(17167)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
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
y_data=to_categorical(y_data)
a=[0.856]
for l in range(1,10):
    seed = 7
    np.random.seed(seed)
    w_train, w_val, y_train1, y_val1=train_test_split(X_data, y_data, test_size=0.2, random_state=7)
    model1 = build_PASNet( shape1=input_shape1,embedding_matrix=embedding_matrix )
    # KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='auto')
    hist = model1.fit(w_train, y_train1, epochs=200, batch_size=64, verbose=1,
                     validation_data=(w_val, y_val1), callbacks=[early_stopping,])
    pred=model1.predict(w_val)
    print(pred.shape)
    accur=acc(y_val1[:,1],pred[:,1])
    a.append(accur)
    if accur>=max(a):
        print('../final_model/model/highway_hs/baseline/PASnet' + motif +str(accur)+'original.h5')
        model1.save('../final_model/model/highway_hs/baseline/PASnet' + motif +str(accur)+'original.h5')
        print('model save')
