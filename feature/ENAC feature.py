import pandas as pd
import numpy as np
from numpy import argmax
from numpy import array
import keras
from tensorflow.keras.utils import to_categorical
from Bio import SeqIO

from tensorflow.keras.utils import to_categorical
from collections import Counter
motif='hg38_hs_AATAAA'


def read_fasta_file():
    fh = open('../split data/train/total/' + motif + '.fa', 'r')


    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            if len(line)>2:
                seq.append(line.replace('\n', '').replace('\r', ''))  # \r\n 一般一起用，用来表示键盘上的回车键，也可只用 \n。
            else:
                continue
    fh.close()

    matrix_data = np.array([list(e) for e in seq],dtype=object)  # 列出每个序列中的核苷酸
    # print(matrix_data)
    # print(len(matrix_data))
    return matrix_data

def ENAC(sequences):
    AA = 'ACGT'
    enac_feature = []
    window = 5
    # for seq in sequences:
    sequences = [x.strip() for x in sequences if x.strip() != '']
    l = len(sequences)
    if l !=606:
        print(l)
        print(sequences)
    enac= []
    for i in range(0, l):
        if i < l and i + window <= l:
            count = Counter(sequences[i:i + window])
            for key in count:
                count[key] = count[key] / len(sequences[i:i + window])
            for aa in AA:
                enac.append(count[aa])

    # enac_feature.append(enac)
    return enac

def feature_extraction(matrix_data):
    final_feature_matrix = [ENAC(e) for e in matrix_data]
    return final_feature_matrix
matrix_data = read_fasta_file()


final_feature_matrix = feature_extraction(matrix_data)

print(np.array(final_feature_matrix,dtype=object).shape)  # (1421, 84)
print(final_feature_matrix[1])
enac=np.array(final_feature_matrix,dtype=object)
np.save('..\ENAC/all/train' + motif + '-ENAC.npy', enac)

print(len(final_feature_matrix[3]))  # (1421, 84)

print(np.array(final_feature_matrix).shape)