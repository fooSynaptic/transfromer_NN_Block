# -*- coding: utf-8 -*-
#/usr/bin/python3


from hyperparams import seq2seq_Hyperparams as hp
import tensorflow as tf
import numpy as np
import os
 
import re
from collections import Counter

#import tokenize
import jieba

def make_vocab(fpath, fname, tokenizer = None):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''  
    #text = codecs.open(fpath, 'r', 'utf-8').read()
    text = open(fpath, 'r', encoding = 'utf-8').readlines()
    text = [line.strip() for line in text if not line.startswith("<")]
    print('length of senteces from path:{} is {}'.format(fpath, len(text)))
    text = ' '.join(text)

    if tokenizer == 'jieba':
        text = re.sub("[\s\p']", "", text)
        words = jieba.cut(text)
    elif tokenizer == None:
        #text = re.sub("[^a-zA-Z]", " ", text)
        words = text.split()
    else:
        raise Exception('Could not find tokenizer...')

    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with open('preprocessed/{}'.format(fname), 'w', encoding = 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))






if __name__ == '__main__':
    make_vocab(hp.source_train, "src.vocab.tsv")
    make_vocab(hp.target_train, "tgt.vocab.tsv", tokenizer = None)
    print("Done")
