
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import numpy as np
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Input, concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

import argparse

import matplotlib.pyplot as plt
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

plt.switch_backend('agg')
motif='AATAAA'
seqLength=606


def DeepGenGrep():
    input_sequence = Input(shape=(seqLength, 4))
    towerA_1 = Conv1D(filters=29, kernel_size=1, padding='same', kernel_initializer='he_normal')(input_sequence)
    towerA_1 = BatchNormalization()(towerA_1)
    towerA_1 = Activation('relu')(towerA_1)
    towerA_2 = Conv1D(filters=121, kernel_size=3, padding='same', kernel_initializer='he_normal')(input_sequence)
    towerA_2 = BatchNormalization()(towerA_2)
    towerA_2 = Activation('relu')(towerA_2)
    towerA_3 = Conv1D(filters=467, kernel_size=5, padding='same', kernel_initializer='he_normal')(input_sequence)
    towerA_3 = BatchNormalization()(towerA_3)
    towerA_3 = Activation('relu')(towerA_3)
    output = concatenate([towerA_1, towerA_2, towerA_3], axis=-1)
    output = MaxPooling1D(pool_size=3, padding='same')(output)
    output = Dropout(rate=0.42198224)(output)

    towerB_1 = Conv1D(filters=216, kernel_size=1, padding='same', kernel_initializer='he_normal')(output)
    towerB_1 = BatchNormalization()(towerB_1)
    towerB_1 = Activation('relu')(towerB_1)
    towerB_2 = Conv1D(filters=237, kernel_size=3, padding='same', kernel_initializer='he_normal')(output)
    towerB_2 = BatchNormalization()(towerB_2)
    towerB_2 = Activation('relu')(towerB_2)
    towerB_3 = Conv1D(filters=517, kernel_size=5, padding='same', kernel_initializer='he_normal')(output)
    towerB_3 = BatchNormalization()(towerB_3)
    towerB_3 = Activation('relu')(towerB_3)
    towerB_4 = Conv1D(filters=458, kernel_size=7, padding='same', kernel_initializer='he_normal')(output)
    towerB_4 = BatchNormalization()(towerB_4)
    towerB_4 = Activation('relu')(towerB_4)
    output = concatenate([towerB_1, towerB_2, towerB_3, towerB_4], axis=-1)
    output = MaxPooling1D(pool_size=3, padding='same')(output)
    output = Dropout(rate=0.53868208)(output)

    output = Conv1D(filters=64, kernel_size=1, kernel_initializer='he_normal')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = LSTM(units=123, return_sequences=True)(output)
    output = Dropout(rate=0.57608335)(output)
    output = LSTM(units=391, return_sequences=True)(output)
    output = Dropout(rate=0.49034301)(output)
    output = Flatten()(output)
    output = Dense(units=1, activation='sigmoid')(output)

    model = Model(input_sequence, output)

    return model
path4='../one hot/train' + motif + '-onehot.npy'
onehot=np.load(path4,allow_pickle=True)
onehot = np.reshape(onehot,(onehot.shape[0],606,4))
# BPB_onehot=np.concatenate((BPB2,onehot),axis=-1)
input_shape3=(onehot.shape[1], onehot.shape[2])


num = int(onehot.shape[0] / 2)
# negy = np.zeros(15540)
# posy = np.ones(17167)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)
onehot=onehot.astype(float)
a=[]
for l in range(1,3):
    seed = 7
    np.random.seed(seed)
    x_train, x_val, y_train, y_val = train_test_split(onehot, y_data, test_size=0.2, random_state=7)
    model = DeepGenGrep( )
    print('Compiling model...')
    model.compile(loss='binary_crossentropy',  # binary_crossentropy / categorical_crossentropy
                  optimizer='nadam',
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='./log/DeepGenGrep')
    # checkpoint = ModelCheckpoint(filepath=f"../baseline/deepgengrep.h5",
    #                              monitor='val_accuracy',
    #                              save_best_only=True,
    #                              save_weights_only=True,
    #                              mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   patience=30,
                                   mode='max')

    callback_lists = [ early_stopping]
    # print('-' * 100)
    hist = model.fit(x_train, y_train,
                     batch_size=64,
                     epochs=150,
                     verbose=1,
                     callbacks=callback_lists,validation_data=(x_val, y_val)
                     )

    # KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred=model.predict(x_val)
    accur=acc(y_val,pred)
    a.append(accur)
    if accur>=max(a):
        model.save('../final_model/model/highway_hs/baseline/deepgengrep'+str(motif)+str(accur)+'.h5')
        print('model save')