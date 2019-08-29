# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function
from hyperparams import feature_Block_Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import re
from jieba import cut

 
def load_vocabs():
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/vocabs.txt', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt] #raw code is hp.mincnt
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word



def create_data(corpus, labels): 
    word2idx, idx2word = load_vocabs()

    one_hot = {0: [1, 0], 1:{0, 1}}
    #max token numbers
    max_token_num = len(word2idx.keys()) + 100
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for sent, label in zip(corpus, labels):
        x = [word2idx.get(word, max_token_num) for word in (sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = one_hot[label]

        if len(x) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))

    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.array(y_list)

    for i, x in enumerate(x_list):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        
    return X, np.array(labels), corpus, labels


def load_train_data(tokenizer = None):
    def _refine(line):
        #line = re.sub("[^\s\p{Latin}']", "", line) 
        return line.strip().split()
    
    if tokenizer == None:
        corpus = [_refine(line) for line in codecs.open(hp.trainset, 'r', 'utf-8').readlines()]
        texts, labels = [' '.join(list(x[0])) for x in corpus], [int(x[1]) for x in corpus]
    #de_sents = [_refine(line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    X, Y, Sources, labels = create_data(texts, labels)
    return X, Y
    
def load_test_data(tokenizer = None):
    def _refine(line):
        #line = re.sub("[^\s\p{Latin}']", "", line) 
        return line.strip().split()
    
    if tokenizer == None:
        corpus = [_refine(line) for line in codecs.open(hp.testset, 'r', 'utf-8').readlines()]
        texts, labels = [' '.join(list(x[0])) for x in corpus], [int(x[1]) for x in corpus]
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

