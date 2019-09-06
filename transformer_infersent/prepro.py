# -*- coding: utf-8 -*-
#/usr/bin/python3

from hyperparams import infersent_Block_Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
 
#import regex
import re
from collections import Counter

#import tokenize
import jieba

def make_vocab(fpath, fname):
    '''Constructs vocabulary.
    
    Args:
      fpath: A list. Input file paths.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    ''' 
    texts = []
    for path in fpath:
        text = [x.strip() for x in codecs.open(path, 'r', 'utf-8').readlines()]
        texts.extend(text)

    corpus = ''.join(texts)
    corpus = re.sub("[\s\p']", "", corpus)
    #replace numbers with NUM
    corpus = re.sub(r'[0-9]+', 'n', corpus)
    corpus = re.sub(r'[a-zA-Z]+', 'α', corpus)
    #words = jieba.cut(corpus)
    words = list(corpus)

    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))




if __name__ == '__main__':
    make_vocab([hp.s1, hp.s2], "vocabs.txt")
    print("Done")
