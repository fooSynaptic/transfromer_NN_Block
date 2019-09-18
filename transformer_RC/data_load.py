# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function
from hyperparams import rc_Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import re
import jieba
import pandas as pd
  
def load_vocabs():
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/vocabs.txt', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt] #raw code is hp.mincnt
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word



def create_data(s1, s2, answer_span):
    """
    the default s1 is the question and s2 is the content
    """
    word2idx, idx2word = load_vocabs()

    
    # Index
    x1_list, x2_list, q_lens, p_lens, s_labels, e_labels,  Questions, Contents = \
    [], [], [], [], [], [], [], []

    for sent1, sent2, span in zip(s1, s2, answer_span):
        x1 = [word2idx.get(word, 1) for word in (sent1 + u" </S>").split()[:hp.q_maxlen-5]] # 1: OOV, </S>: End of Text
        x2 = [word2idx.get(word, 1) for word in (sent2 + u" </S>").split()[:hp.p_maxlen-5]] 

        x1_list.append(np.array(x1))
        x2_list.append(np.array(x2))

        q_lens.append(len(x1))
        p_lens.append(len(x2))

        s_labels.append(span[0])
        e_labels.append(span[1])

    print('Demo:', x1_list[0], x2_list[0], q_lens[0], p_lens[0], s_labels[0], e_labels[0])

    # Pad      
    X1 = np.zeros([len(x1_list), hp.q_maxlen], np.int32)
    X2 = np.zeros([len(x2_list), hp.p_maxlen], np.int32)

    for i, x in enumerate(x1_list):
        X1[i] = np.lib.pad(x, [0, hp.q_maxlen-len(x)], 'constant', constant_values=(0, 0))

    for i, x in enumerate(x2_list):
        X2[i] = np.lib.pad(x, [0, hp.p_maxlen-len(x)], 'constant', constant_values=(0, 0))


    
    return X1, X2, q_lens, p_lens, s_labels, e_labels


def _refine(line, lan = 'zh'):
    if lan == 'zh':
        line = re.sub("[\s\p']", "", line)
        #line = re.sub(r'[0-9]+', ' n', line)
        #line = re.sub(r'[a-zA-Z]+', ' α', line)
        line = jieba.cut(line)
        return ' '.join(list(line))
    elif lan == 'en':
        line = re.sub("[^a-zA-Z]", " ", line)
        return line

    else:
        raise Exception('Havn\'t specified language!')
        return



def load_train_data(tokenizer = None):
    train_data = pd.read_csv(hp.trainfile)
    questions, contents, answer_spans = list(train_data['question']), list(train_data['content']), \
    list(train_data['answer_span'])

    questions, contents = [_refine(line) for line in questions], [_refine(line) for line in contents]
    
    answer_spans = [eval(line) for line in answer_spans]

    X1, X2, q_lens, p_lens, start_labels, end_labels = create_data(questions, contents, answer_spans)
    return X1, X2, q_lens, p_lens, start_labels, end_labels
    


def load_test_data(tokenizer = None):
    test_data = pd.read_csv(hp.testfile)
    questions, contents, answer_spans = list(test_data['question']), list(test_data['content']), \
    list(test_data['answer_span'])

    answer_spans = [eval(line) for line in answer_spans]

    questions, contents = [_refine(line) for line in questions], [_refine(line) for line in contents]
    

    X1, X2, q_lens, p_lens, start_labels, end_labels = create_data(questions, contents, answer_spans)
    return X1, X2, q_lens, p_lens, start_labels, end_labels


def get_batch_data():
    # Load data
    X1, X2, q_lens, p_lens, start_labels, end_labels = load_train_data()
    
    # calc total batch count
    num_batch = len(X1) // hp.batch_size
    
    # Convert to tensor
    X1 = tf.convert_to_tensor(X1, tf.int32)
    X2 = tf.convert_to_tensor(X2, tf.int32)
    q_lens = tf.convert_to_tensor(q_lens, tf.int32)
    p_lens = tf.convert_to_tensor(p_lens, tf.int32)
    start_labels = tf.convert_to_tensor(start_labels, tf.int32)
    end_labels = tf.convert_to_tensor(end_labels, tf.int32)

    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X1, X2, q_lens, p_lens, start_labels, end_labels])
            
    # create batch queues
    q, p, q_length, p_length, start_pos, end_pos = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64, 
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)

    
    return q, p, q_length, p_length, start_pos, end_pos, num_batch # (N, T), (N, T), ()

