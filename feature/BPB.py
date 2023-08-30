import numpy as np
from Bio import SeqIO
# from nltk import trigrams, bigrams
# from keras.preprocessing.text import Tokenizer
# from gensim.models import Word2Vec
# import re
# from keras.layers import Convolution1D,Embedding, Softmax,RepeatVector, Add, Lambda,MaxPooling1D, Dense,Lambda, Dropout,Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
#
# from keras import backend as K, regularizers
# from keras.datasets import imdb
# from keras import preprocessing
# from keras.models import Sequential
# from keras.layers import Flatten,Dense,Embedding,Convolution1D,Dropout,Activation,MaxPooling1D
# #from keras.optimizers import SGD,Adam
# from keras.models import load_model
# from keras.models import Model
# from keras.utils import np_utils
# from keras.callbacks import TensorBoard
# from sklearn.model_selection import train_test_split
# from keras.callbacks import ReduceLROnPlateau
# import matplotlib.pyplot as plt

# pos=[]
# for index1, record1 in enumerate(SeqIO.parse('D:\论文\数据\PAS data seq\CD-HIT 0.8/positive/TATAAA-0.8.fa', 'fasta')):
#     pos.append(record1)
# neg=[]
# for index2, record2 in enumerate(SeqIO.parse('D:\论文\数据\PAS data seq\CD-HIT 0.8/negative/neg-TATAAA-0.8.fa', 'fasta')):
#     neg.append(record2)
# print(len(pos))
# print(len(neg))
# AA='ACGT'
# lens=len(pos[1])
# print(lens)
# F1=np.zeros((4,lens))
# F2=np.zeros((4,lens))
# print(F1.shape,F2.shape)
# for m in range(len(pos)):
#     for i in range(lens):
#         t=pos[m][i]
#         k=AA.find(t)
#         F1[k,i]= F1[k,i]+1
#
# for m in range(len(neg)):
#     for i in range(lens):
#         t=neg[m][i]
#         k=AA.find(t)
#         F2[k,i]= F2[k,i]+1
# F1=F1/len(pos)
# F2=F2/len(neg)
# print(F1)
# print(F2)
# BPBpos=np.zeros((len(pos),2*lens))
# BPBneg=np.zeros((len(neg),2*lens))
# print(BPBpos.shape,BPBneg.shape)
# for m in range(len(pos)):
#     for i in range(lens):
#         t=pos[m][i]
#         k=AA.find(t)
#         BPBpos[m,i]=F1[k,i]
#         BPBpos[m,i+lens]=F2[k,i]
# print(BPBpos.shape)
# for m in range(len(pos)):
#     for i in range(lens):
#         t=neg[m][i]
#         k=AA.find(t)
#         BPBneg[m,i]=F1[k,i]
#         BPBneg[m,i+lens]=F2[k,i]
# print(BPBneg.shape)
# BPB=np.concatenate((BPBneg,BPBpos),axis=0)
# np.save('D:\论文\数据\PAS data seq\CD-HIT 0.8/TATAAA-BPB',BPB)

def BPB(seq):
    pos = []
    for index1, record1 in enumerate(SeqIO.parse('../AATAAA/H AATAAA/hs_AATAAA_polyA.fa', 'fasta')):
        pos.append(record1)
    neg = []
    for index2, record2 in enumerate(
            SeqIO.parse('../AATAAA/H AATAAA/hs_negAATAAA_polyA.fa', 'fasta')):
        neg.append(record2)
    # print(len(pos))
    # print(len(neg))
    AA = 'ACGT'
    lens = len(pos[1])
    # print(lens)
    F1 = np.zeros((4, lens))
    F2 = np.zeros((4, lens))
    # print(F1.shape, F2.shape)
    for m in range(len(pos)):
        for i in range(lens):
            t = pos[m][i]
            k = AA.find(t)
            F1[k, i] = F1[k, i] + 1

    for m in range(len(neg)):
        for i in range(lens):
            t = neg[m][i]
            k = AA.find(t)
            F2[k, i] = F2[k, i] + 1
    F1 = F1 / len(pos)
    F2 = F2 / len(neg)
    # print(F1)
    # print(F2)
    BPBpos = np.zeros((len(pos), 2 * lens))
    BPBneg = np.zeros((len(neg), 2 * lens))
    print(BPBpos.shape, BPBneg.shape)
    for m in range(len(pos)):
        for i in range(lens):
            t = pos[m][i]
            k = AA.find(t)
            BPBpos[m, i] = F1[k, i]
            BPBpos[m, i + lens] = F2[k, i]
    print(BPBpos.shape)
    for m in range(len(pos)):
        for i in range(lens):
            t = neg[m][i]
            k = AA.find(t)
            BPBneg[m, i] = F1[k, i]
            BPBneg[m, i + lens] = F2[k, i]
    print(BPBneg.shape)
    BPB = np.concatenate((BPBneg, BPBpos), axis=0)
    # np.save('D:\论文\数据\PAS data seq\CD-HIT 0.8/' + str(seq) + '-BPB', BPB)
    # np.save('D:\data\Kalkataw data model\BPB/neg/'+ str(seq) + '-BPB',BPBneg)
    # np.save('D:\data\Kalkataw data model\BPB/pos/'+ str(seq) + '-BPB',BPBpos)

    np.save('D:\data\Kalkataw data model\BPB/H' + str(seq) + '-BPB', BPB)

    return   BPB

seq='AATAAA'

a=BPB(seq)
print(a)