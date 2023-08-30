


from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D, Convolution2D,Conv2D
from keras.utils import np_utils
import numpy as np
from keras import backend as k
# k.set_image_dim_ordering('th')

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import scipy.io as spio
import os.path
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

motif='AATAAA'
import matplotlib.pyplot as plt

nb_classes=2
nb_epoch = 100
val_split = 0.2
pool_size = (1,2)
batch_size = 16
optimizer = 'Adadelta'
init_mode = 'zero'
activation = 'tanh'
filter = 50
filterL = 30
filterW = 32
neurons = 32
dropout_rate = 0.1
tri_path='../one hot/tritrainAATAAA_onehot.npy'
trionehot=np.load(tri_path)
channel=1
rows=trionehot.shape[1]
cols=trionehot.shape[2]
trionehot = np.reshape(trionehot,(trionehot.shape[0],1,598,64))
# BPB_onehot=np.concatenate((BPB2,onehot),axis=-1)
input_shape3=(channel,rows,cols)
print(input_shape3)
num = int(trionehot.shape[0] / 2)
# negy = np.zeros(15540)
# posy = np.ones(17167)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)
trionehot=trionehot.astype(float)



model=Sequential()

model = Sequential()

# 添加第一层卷积 + 最大池化层
model.add(Conv2D(50, (30, 32), padding='same', activation='relu', kernel_initializer='zeros', input_shape=(1, 598, 64)))
model.add(MaxPooling2D(pool_size=(1, 2)))

# 添加第二层卷积层
model.add(Conv2D(100, (10, 8), padding='same', activation='relu', kernel_initializer='glorot_uniform'))

model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(dropout_rate))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation(activation))
model.add(Dropout(dropout_rate))

model.add(Dense(1))
model.add(Activation('softmax'))

model.summary()
model.get_config()
# Compile model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
a=[]
for l in range(1,3):
    seed = 7
    np.random.seed(seed)
    x_train, x_val, y_train, y_val = train_test_split(trionehot, y_data, test_size=0.2, random_state=7)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    hist = model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(x_val, y_val),
                     callbacks=[early_stopping])




    # KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred=model.predict(x_val)
    accur=acc(y_val,pred)
    a.append(accur)
    if accur>=max(a):
        model.save('../final_model/model/highway_hs/baseline/deepgsr'+str(motif)+str(accur)+'.h5')
        print('model save')
