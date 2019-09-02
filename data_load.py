# -*- coding: utf-8 -*-
# /usr/bin/python3

from hyperparams import seq2seq_Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import re
import jieba
from bs4 import BeautifulSoup as bs

 
def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt] 
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_zh_vocab():
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/zh.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    en2idx, idx2en = load_en_vocab()
    zh2idx, idx2zh = load_zh_vocab()
    #max token numbers
    max_token_num = max(len(en2idx.keys()), len(zh2idx.keys())) + 100
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        #the default source senteces is english and target sentences is chinese
        x = [en2idx.get(word, max_token_num) for word in source_sent.split()[:hp.maxlen-5] + [u" </S>"]]
        y = [zh2idx.get(word, max_token_num) for word in target_sent.split()[:hp.maxlen-5] + [u" </S>"]]
        
        x_list.append(np.array(x))
        y_list.append(np.array(y))
        Sources.append(source_sent)
        Targets.append(target_sent)
    print('Demo: {}->\n{}'.format(Sources[0], Targets[0]))
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        #print(x, y, hp.maxlen, len(x), len(y))
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets


def refine(line, tokenizer):
    if tokenizer == 'jieba':
        line = re.sub("[\s\p']", "", line)
        return ' '.join(jieba.cut(line))
    elif tokenizer == 'en':
        line = re.sub("[^a-zA-Z]", " ", line)
        return line
    else:
        raise Exception('Could not find tokenizer...') 

def load_train_data():    
    en_sents = [refine(line, tokenizer = 'en') \
        for line in open(hp.source_train, 'r', encoding = 'utf-8').read().split("\n") \
            if not line.startswith('<')]
    zh_sents = [refine(line, tokenizer = 'jieba') \
        for line in open(hp.target_train, 'r', encoding = 'utf-8').read().split("\n") \
            if not line.startswith('<')]

    X, Y, Sources, Targets = create_data(en_sents, zh_sents)
    return X, Y
    
def load_test_data():
    def _parser(text):
        return [x.text for x in bs(text).find_all('seg')]

    '''
    en_sents = [refine(line, tokenizer = 'en') \
        for line in open(hp.source_test, 'r', encoding = 'utf-8').read().split("\n") \
            if line.startswith('<seg id')]
    zh_sents = [refine(line, tokenizer = 'jieba') \
        for line in open(hp.target_test, 'r', encoding = 'utf-8').read().split("\n") \
            if line.startswith('<seg id')]
    '''
    en_sents = [refine(line, tokenizer = 'en') for line in _parser(open(hp.source_test).read().strip())]
    zh_sents = [refine(line, tokenizer = 'jieba') for line in _parser(open(hp.target_test).read().strip())]

    X, Y, Sources, Targets = create_data(en_sents, zh_sents)
    return X, Sources, Targets # (1064, 150)

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

