import pandas as pd
import numpy as np
from numpy import argmax
from numpy import array
import keras
from tensorflow.keras.utils import to_categorical
from Bio import SeqIO

motif='hg38_hs_AATAAA'
def read_fasta_file():
    fh = open('../split data/test/total/' + motif + '.fa', 'r')


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


def extract_line(data_line):
    A = [0, 0, 0, 1]
    T = [0, 0, 1, 0]
    C = [0, 1, 0, 0]
    G = [1, 0, 0, 0]
    # A = [0.126]
    # C = [0.134]
    # G = [0.0806]
    # T = [0.1335]

    feature_representation = {"A": A, "C": C, "G": G, "T": T}
    one_line_feature = []
    for index, data in enumerate(data_line):
        # enumerate( sequence, [start = 0]),其中，第一个参数是一个可迭代的序列，第二个参数start则是序列中首个元素的行号从几开始（默认从0开始）
        # enumerate()函数返回的是enumerate对象实例，他是一个迭代器，可返回连续的元组。元组由索引值和对传入的序列不断调用next()方法得到的值组成。
        # print(index, data)

        if data in feature_representation.keys():
            # print(index, data)  # 每一个序列的每个核苷酸代表一个索引，为 0-40（0 A……40 T）
            # print(feature_representation[data])  # 每个核苷酸的编码#[1, 0, 0, 0]#[0, 1, 0, 0]
            one_line_feature.extend(feature_representation[data])
    return one_line_feature


def feature_extraction(matrix_data):
    final_feature_matrix = [extract_line(e) for e in matrix_data]
    return final_feature_matrix


matrix_data = read_fasta_file()
# for e in matrix_data:
#     final_feature_matrix=extract_line(e)
# print(matrix_data.shape)  # 将数据用矩阵表示出来
# print(len(final_feature_matrix))
final_feature_matrix = feature_extraction(matrix_data)
print(len(final_feature_matrix))
print(type(final_feature_matrix))
# # print(final_feature_matrix)   #将所有排成一列
#
# EIIP=np.array(final_feature_matrix,dtype=object)
# np.save(".\EIIP/testAATAAA-EIIp.npy", EIIP)
Onehot=np.array(final_feature_matrix,dtype=object)
np.save('../one hot/all/test'+ motif+'-onehot.npy', Onehot)
print(np.array(final_feature_matrix).shape)  # (1421, 84)

# #
# # pd.DataFrame(final_feature_matrix).to_csv('D:\code\PAS feature\one hot/TATAAA-0.8.csv',header=None,index=False)
#
# # csv_file=csv.reader(open('D:\论文\数据\PAS data seq/all seq\one hot/AGTAAA.csv','r'))
#
# final_feature_matrix1 = np.array(final_feature_matrix,dtype=object)


# np.save("D:\code\PAS feature\one hot/TATAAA-0.8.npy", final_feature_matrix1)
# # 下载矩阵
# X = np.load("D:\code\PAS feature\one hot/TATAAA-0.8.npy",allow_pickle=True)



