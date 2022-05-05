# -*- coding: utf-8 -*-
# @Author  : clq
# @FileName: tools.py
# @Software: PyCharm
import os

import gensim.models
import numpy as np
from pathlib import Path
from keras.models import load_model
from tools import supple_X,read_fasta
import argparse
from model import ourmodel
#定义函数
def Gen_Words(sequences,kmer_len,s):
    out=[]
    for i in sequences:
        kmer_list=[]
        for j in range(0,(len(i)-kmer_len)+1,s):
            kmer_list.append(i[j:j+kmer_len])
        out.append(kmer_list)
    return out

def ArgsGet():
    parse = argparse.ArgumentParser(description='NeuroPred-CLQ')
    parse.add_argument('--file', type=str, default='test.fasta', help = 'fasta file')
    parse.add_argument('--outfile',type=str, default='supple_test.fasta', help = 'fasta file')
    parse.add_argument('--out_path', type=str, default='result', help='output path')
    args = parse.parse_args()
    return args

def process_data(file,outfile):
    supple_X(file,outfile,100)
    seq_data = read_fasta(outfile)
    data = seq_data.iloc[:,1].to_numpy()

    return data

def predict(model,data,output_path):
    model.load_weights('CLQ_model/FinModel.h5')
    y_p = model.predict([data])
    output_file = os.path.join(output_path, 'result.txt')
    np.savetxt(output_file,y_p[:,1])

if __name__ == '__main__':
    args = ArgsGet()
    file = args.file
    outfile = args.outfile
    output_path = args.out_path
    # building output path directory
    Path(output_path).mkdir(exist_ok=True)

    #reading file
    data = process_data(file,outfile)
    W_model = gensim.models.Word2Vec.load('CLQ_model/NPs4')
    x_test3 = Gen_Words(data, 4, 1)
    X_test = []
    for i in range(0, len(x_test3)):
        s = []
        for word in x_test3[i]:
            if word in W_model.wv:
                s.append(W_model.wv[word])
            else:
                s.append(np.zeros([150, ]))
        X_test.append(s)
    X_test = np.array(X_test)
    model = ourmodel()
    predict(model,X_test,output_path)





