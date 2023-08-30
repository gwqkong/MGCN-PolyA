import pandas as pd
import sys
import time
import numpy as np
import pandas as pd
from Bio import  SeqIO


sixseq=['AAAAAG','AAGAAA','AATAAA','AATACA','AATAGA','AATATA','ACTAAA','AGTAAA','ATTAAA','CATAAA','GATAAA','TATAAA']
for i in range(len(sixseq)):
    all='D:\论文\数据\PAS data seq\CD-HIT 0.8/all/'+str(sixseq[i])+'-0.8.fa'
    fall=open(all,'w+')
    neg='D:\论文\数据\PAS data seq\CD-HIT 0.8/negative/neg-'+str(sixseq[i])+'-0.8.fa'
    pos='D:\论文\数据\PAS data seq\CD-HIT 0.8/positive/'+str(sixseq[i])+'-0.8.fa'
    fn=open(neg,'r+')
    fp=open(pos,'r+')
    lines1=fn.readlines()
    lines2=fp.readlines()
    for k in lines1:
        fall.write(k)
    for j in lines2:
        fall.write(j)
    fall.close()
    fn.close()
    fp.close()
