# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:19:53 2020

@author: wanghongfei
"""

import numpy as np
from Bio import SeqIO
from nltk import trigrams, bigrams,fivegrams
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re


# np.set_printoptions(threshold=np.inf)#np.set_printoptions()用于控制Python中小数的显示精度
# # fa=open('D:\论文\数据\human PAS/all data/ATTAAA.fa','r+')
# # lines=fa.readlines()
# # with open('D:\论文\数据\PAS data seq/all seq/', 'w+') as f:
# #     f.write(open('D:\论文\数据\PAS data seq/negative seq/', 'r').read())
# #     f.write(open('D:\论文\数据\PAS data seq\positive seq/', 'r').read())
# texts = []
# for index, record in enumerate(SeqIO.parse('D:\论文\数据\PAS data seq\CD-HIT 0.8/all/TATAAA-0.8.fa','fasta')):#enumerate() 这个函数的基本应用就是用来遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置。
#     # tri_tokens = bigrams(record)
#     tri_tokens = trigrams(record)
#     temp_str = ""
#     for item in ((tri_tokens)):
#         #print(item),
#         temp_str = temp_str + " " +item[0] + item[1] +item[2]
#         # temp_str = temp_str + " " + item[0] + item[1]
#         #temp_str = temp_str + " " +item[0]
#     texts.append(temp_str)
# #print(texts)
# seq=[]
# stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
# for doc in texts:
#     doc = re.sub(stop, '', doc)
#     seq.append(doc.split())# re.sub() 的详细用法，该函数主要用于替换字符串中的匹配项
# print(seq)
#
# w2v_model = Word2Vec(seq, vector_size=3,window=5, min_count=1, workers=4, sg=1)
#
# vocab_list = list(w2v_model.wv.key_to_index.keys())
#
# word_index = {word: index for index, word in enumerate(vocab_list)}
#
# w2v_model.save('./model/TATAAA-3mer-0.8_word2vec_3.model')
# # #
# # #
# w2v_model.wv.save_word2vec_format('./model/TATAAA-3mer-0.8_word2vec_3.vector')#词向量
# # fa.close()



def five_mers(seq):
    """
    利用滑动窗口将一个序列拆分成5mer
    :param seq: 输入的序列（字符串或列表等可迭代对象）
    :return: 含有所有5mer的列表
    """
    k = 5
    return [seq[i:i+k] for i in range(len(seq)-(k-1))]
motif='hg38_hs_AATAAA'
for i in [90]:
    np.set_printoptions(threshold=np.inf)  # np.set_printoptions()用于控制Python中小数的显示精度
    # fa=open('D:\论文\数据\human PAS/all data/ATTAAA.fa','r+')
    # lines=fa.readlines()
    # with open('D:\论文\数据\PAS data seq/all seq/', 'w+') as f:
    #     f.write(open('D:\论文\数据\PAS data seq/negative seq/', 'r').read())
    #     f.write(open('D:\论文\数据\PAS data seq\positive seq/', 'r').read())
    texts = []
    # for index, record in enumerate(SeqIO.parse('../all PAS data/Kalkataw bt/bt_all_polyA.fa', 'fasta')):  # enumerate() 这个函数的基本应用就是用来遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置。
    for index, record in enumerate(SeqIO.parse('../AATAAA/H AATAAA/H AATAAA.fa',
                                               'fasta')):  # enumerate() 这个函数的基本应用就是用来遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置。
    #     tri_tokens = bigrams(record)
    #     tri_tokens = trigrams(record)
        tri_tokens = fivegrams(record)
        temp_str = ""
        for item in ((tri_tokens)):
            # print(item),
            temp_str = temp_str + " " + item[0] + item[1] + item[2]+item[3]+item[4]
            # temp_str = temp_str + " " + item[0] + item[1]
            # temp_str = temp_str + " " +item[0]
        texts.append(temp_str)
    # print(texts)
    seq = []
    stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for doc in texts:
        doc = re.sub(stop, '', doc)
        seq.append(doc.split())  # re.sub() 的详细用法，该函数主要用于替换字符串中的匹配项
    print(seq)

    w2v_model = Word2Vec(seq, vector_size=i, window=5, min_count=1, workers=4, sg=1)

    vocab_list = list(w2v_model.wv.key_to_index.keys())

    word_index = {word: index for index, word in enumerate(vocab_list)}
    i=str(i)

    w2v_model.save('..\model\Kalkataw hs/PASnet ' + motif + '_word2vec_'+i+'.model')
    # # #
    # # #
    w2v_model.wv.save_word2vec_format('..\model\Kalkataw hs/PASnet ' + motif + '_word2vec_'+i+'.vector')  # 词向量
    # w2v_model.save('D:\data/big human model/AATAAA-2mer' + i + '.model')
    # # #
    # # #
    # w2v_model.wv.save_word2vec_format('D:\data/big human model/AATAAA-2mer' + i + '.vector')  # 词向量
    # # fa.close()
    # from gensim import models
    #
    # w2v_model =models.KeyedVectors.load_word2vec_format('./model/Kalkataw hs/H AATAAA_word2vec_'+i+'.model')
