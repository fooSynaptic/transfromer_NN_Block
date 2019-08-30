# -*- coding: utf-8 -*-
#/usr/bin/python3

import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import feature_Block_Hyperparams as hp
from data_load import load_vocabs, load_train_data, load_test_data, create_data
from train import Graph
from nltk.translate.bleu_score import corpus_bleu
import argparse




def cut(seq, label):
    if isinstance(seq, str):
        seq = seq.split()
    if isinstance(label, str):
        label = label.split()

    seq = seq + ['PAD']*(len(label) - len(seq))
    assert len(seq) == len(label), "seq label is not compliable...{}, {}".format(seq, label)
    tokens = []
    i = 0
    while i < len(seq):
        if label[i] == 'S':
            tokens.append(seq[i])
        elif label[i] == 'B':
            tmp = seq[i]
            while i+1 < len(seq) and label[i+1] == 'M':
                tmp += seq[i+1]
                i += 1
            if not i+1 < len(seq): break
            #print(label[i+1], seq[i+1])
            if label[i+1] == 'E':
                tmp += seq[i+1]
            tokens.append(tmp)
        i += 1
    return ' '.join(tokens)

#test case
print(cut('l i k e m e','B M M E B E'))






def eval(task_name):
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Y, Texts, Labels = load_test_data()

    word2idx, idx2word = load_vocabs()

    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            print('Model dir:', hp.logdir)
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            print("Model name:", mname)
             
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                print("Iterator:", len(X), hp.batch_size)
                for i in range(len(X) // hp.batch_size):                
                    print('Step:\t', i)     
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sentences = Texts[i*hp.batch_size: (i+1)*hp.batch_size]
                    labels = Labels[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    

                    ### Write to file
                    for sent, label, pred in zip(Texts, Labels, preds): # sentence-wise
                        #print('Inspecting:', source, target, pred)
                        #got = " ".join(idx2word[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- sent: " + sent +"\n")
                        fout.write('- label: {}, -predict: {} \n'.format(label, pred))
                        fout.flush()
                        
                        # bleu score
                        if task == 'seq2seq':
                            ref = target.split()
                            hypothesis = got.split()
                            if len(ref) > 3 and len(hypothesis) > 3:
                                list_of_refs.append([ref])
                                hypotheses.append(hypothesis)
                                 

                ## Calculate bleu score
                if task == 'seq2seq':
                    score = corpus_bleu(list_of_refs, hypotheses)
                    fout.write("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choice the task you want to eval.')
    parser.add_argument('--task', help='task name(default: classfication)')

    args = parser.parse_args()
    task_name = args.task
    eval(task_name)
    print("Done")
    

