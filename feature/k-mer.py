from itertools import product
import numpy  as np
def nucleotide_type(k):
    z = []
    for i in product('ACGT', repeat = k):  #笛卡尔积（有放回抽样排列）
        z.append(''.join(i))  #把('A,A,A')转变成（AAA）形式
    return z
# 碱基对数量统计
def char_count(sequence,num,k):
    n = 0
    char = nucleotide_type(k)  #调用提取核苷酸类型模块
    for i in range(len(sequence)-k+1):   #统计相应字符出现的数量
        if sequence[i:i+k] == char[num]:
            n += 1
    return n/(len(sequence)-k+1)  #返回频率（出现的次数/总次数）总次数=序列长度-取几个碱基+1
def feature(seq,k):
    list = []
    for i in range(4**k):   #根据核苷酸类型数量来取值（二、三、四核苷酸分别循环16、64、256次）
        list.append(char_count(seq,i,k))
    return (list)
# 逐行调用特征编码
def Sequence_replacement(sequ,k):
    sequen = [None]*len(sequ)
    for i in range(len(sequ)):
        s = sequ[i]
        sequen[int(i)] = feature(s,k)
    return sequen

motif='hg38_hs_AATAAA'

fh = open('..\split data/test/total/' + motif + '.fa', 'r')


data = []
for line in fh:
   if line.startswith('>'):
            continue
   else:
            if len(line)>2:
                data.append(line.replace('\n', '').replace('\r', ''))  # \r\n 一般一起用，用来表示键盘上的回车键，也可只用 \n。
            else:
                continue
fh.close()
print(len(data))

k1=1
k2=2
k3=3
feature_1nf = Sequence_replacement(data,k1)  #data为具体数据，k的值根据需要自己设定
feature_2nf = Sequence_replacement(data,k2)
feature_3nf=Sequence_replacement(data,k3)
feature_1nf=np.asarray(feature_1nf)
feature_2nf=np.asarray(feature_2nf)
feature_3nf=np.asarray(feature_3nf)
k_mer=np.concatenate((feature_1nf,feature_2nf,feature_3nf),axis=1)
print(k_mer.shape)
np.save(r'..\K_mer/all/test'+motif+'.npy',arr=k_mer)