# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function
from hyperparams import infersent_Block_Hyperparams as hp
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



def create_data(s1, s2, labels): 
    word2idx, idx2word = load_vocabs()

    #max token numbers
    max_token_num = len(word2idx.keys()) + 100
    
    # Index
    x1_list, x2_list, Sources, Targets = [], [], [], []
    for sent1, sent2 in zip(s1, s2):
        x1 = [word2idx.get(word, 1) for word in (sent1 + u" </S>").split()[:hp.maxlen-5]] # 1: OOV, </S>: End of Text
        x2 = [word2idx.get(word, 1) for word in (sent1 + u" </S>").split()[:hp.maxlen-5]] 

        x1_list.append(np.array(x1))
        x2_list.append(np.array(x2))

    # Pad      
    X1 = np.zeros([len(x1_list), hp.maxlen], np.int32)
    X2 = np.zeros([len(x2_list), hp.maxlen], np.int32)

    for i, x in enumerate(x1_list):
        X1[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))

    for i, x in enumerate(x2_list):
        X2[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))

        
    return X1, X2, np.array(labels)


def _refine(line):
    #line = re.sub("[^\s\p{Latin}']", "", line) 
    line = re.sub("[\s\p']", "", line)
    line = re.sub(r'[0-9]+', 'n', line)
    line = re.sub(r'[a-zA-Z]+', 'α', line)
    return ' '.join(list(line))


def load_train_data(tokenizer = None):
    if tokenizer == None:
        corpus = [line.strip().split('<>') for line in codecs.open(hp.trainset, 'r', 'utf-8').readlines()]
        s1, s2, labels = [_refine(line[1]) for line in corpus], [_refine(line[2]) for line in corpus], \
        [line[0] for line in corpus]

    X1, X2, Label = create_data(s1, s2, labels)
    return X1, X2, Label
    


def load_test_data(tokenizer = None):
    if tokenizer == None:
        corpus = [line.strip().split('<>') for line in codecs.open(hp.testset, 'r', 'utf-8').readlines()]
        s1, s2, labels = [_refine(line[1]) for line in corpus], [_refine(line[2]) for line in corpus], \
        [line[0] for line in corpus] 

    X1, X2, Label = create_data(s1, s2, labels)
    return X1, X2, Label


def get_batch_data():
    # Load data
    X1, X2, Label = load_train_data()
    
    # calc total batch count
    num_batch = len(X1) // hp.batch_size
    
    # Convert to tensor
    X1 = tf.convert_to_tensor(X1, tf.int32)
    X2 = tf.convert_to_tensor(X2, tf.int32)
    Label = tf.convert_to_tensor(Label, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X1, X2, Label])
            
    # create batch queues
    x1, x2, label = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64, 
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x1, x2, label, num_batch # (N, T), (N, T), ()

