# -*- coding: utf-8 -*-
# @Author  : clq
# @FileName: tools.py
# @Software: PyCharm

from Bio import SeqIO
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')


# 读取文件
def read_fasta(fname):
    with open(fname, "rU") as f:
        seq_dict = [(record.id, record.seq._data.decode())
                    for record in SeqIO.parse(f, "fasta")]
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df


# 对长短不一的序列补充X
'''
主要有三部分
1、读取数据
构造字典形式
{'amp5_30_1': 'ACSAG',
 'amp5_30_2': 'AMVGT',
 'amp5_30_3': 'AMVSS',
 'amp5_30_4': 'CPFVC'}

2、填补X
如果序列长度大于规定长度要进行裁剪，设置裁剪规则，从百分之多少进行裁剪
{'amp5_30_1': 'ACSAGXXXXXXXXXXXXXXXXXXXXXXXXX',
 'amp5_30_2': 'AMVGTXXXXXXXXXXXXXXXXXXXXXXXXX',
 'amp5_30_3': 'AMVSSXXXXXXXXXXXXXXXXXXXXXXXXX',
 'amp5_30_4': 'CPFVCXXXXXXXXXXXXXXXXXXXXXXXXX',}
 3、保存文件
'''


def read_data(Filename):
    dictin = {}
    i = 1
    with open(Filename) as fid:
        name = None
        for line in fid:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                name = line.strip()[1:] + str(i)
                dictin[name] = ''
                i += 1
            else:
                if name is None:
                    continue
                dictin[name] += line.strip()
    return dictin


def fixOneSeq(seqIn, fixFrontScale, cutFrontScale, spcLen, paddingRes='X'):
    if len(seqIn) > spcLen:
        # cut
        exceedLen = len(seqIn) - spcLen
        frontLen = int(np.rint(float(exceedLen) * cutFrontScale))
        # lastLen = exceedLen - frontLen
        outSeq = seqIn[frontLen:frontLen + spcLen]
    elif len(seqIn) < spcLen:
        #        print(len(seqIn) , spcLen)
        # add
        exceedLen = spcLen - len(seqIn)
        frontLen = int(np.rint(float(exceedLen) * fixFrontScale))
        lastLen = exceedLen - frontLen
        outSeq = ''
        outSeq += paddingRes * frontLen
        outSeq += seqIn
        outSeq += paddingRes * lastLen
    #        print(outSeq)
    else:
        outSeq = seqIn
    return outSeq


def printout(fileout, Dictin):
    with open(fileout, 'w') as FIDO:
        for k in Dictin:
            FIDO.write('>%s\n' % k)
            tmpstr = Dictin[k]
            FIDO.write('%s\n' % tmpstr)


# 对代码进行整合
def supple_X(in_Filename, out_Filename, maxl):
    Dictin = read_data(in_Filename)
    outDict = {}
    for k in Dictin:
        tmpOut = fixOneSeq(Dictin[k], 0, 0, maxl, paddingRes='X')
        outDict[k] = tmpOut
    printout(out_Filename, outDict)