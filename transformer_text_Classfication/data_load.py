# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function
from hyperparams import feature_Block_Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import re
from jieba import cut
from collections import Counter
  
tagging = {'时尚':0, '教育':1, '时政':2, '体育':3, '游戏':4, '家居':5, '科技':6, '房产':7, '财经':8, '娱乐':9}


def load_vocabs():
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/vocabs.txt', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt] #raw code is hp.mincnt
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word



def create_data(corpus, labels): 
    word2idx, idx2word = load_vocabs()


    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for sent, label in zip(corpus, labels):
        x = [word2idx.get(word, 1) for word in (sent + u" </S>").split()[:hp.maxlen]] # 1: OOV, </S>: End of Text
        x_list.append(np.array(x))


    # Pad 
    X = np.zeros([len(x_list), hp.maxlen], np.int32)

    for i, x in enumerate(x_list):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        
    return X, np.array(labels), corpus, labels



def _refine(line):
	line = re.sub("[\s\p']", "", line)
	line = re.sub('[0-9]+', 'N', line)
	line = re.sub('[a-zA-Z]+', 'α', line)
	return ' '.join(list(line))


def load_train_data(tokenizer = None):
    if tokenizer == None:
        corpus = [line.strip().split() for line in codecs.open(hp.trainset, 'r', 'utf-8').readlines()]
        corpus = [line for line in corpus if line[0] in tagging]
        texts, labels = [_refine(line[1]) for line in corpus], [tagging[line[0]] for line in corpus]
    
    X, Y, Sources, labels = create_data(texts, labels)
    return X, Y
    
def load_test_data(tokenizer = None):
    if tokenizer == None:
        corpus = [line.strip().split() for line in codecs.open(hp.testset, 'r', 'utf-8').readlines()]
        corpus = [line for line in corpus if line[0] in tagging]
        texts, labels = [_refine(line[1]) for line in corpus], [tagging[line[0]] for line in corpus] 

    X, Y, Sources, labels = create_data(texts, labels)
    return X, Y, Sources, labels


def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    X, Y 
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64, 
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()

