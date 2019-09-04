# -*- coding: utf-8 -*-
#/usr/bin/python3

import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import seq2seq_Hyperparams as hp
from data_load import load_en_vocab, load_zh_vocab, load_train_data, load_test_data, create_data
from train import Graph

import argparse
from modules import bleu
import math
from modules import cut




def eval(task_name):
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Sources, Targets = load_test_data()
    #print(X, Sources, Targets)
    en2idx, idx2en = load_en_vocab()
    zh2idx, idx2zh = load_zh_vocab()
     
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            print('Model dir:', hp.logdir)
            mname = '{}'.format(''.join(hp.source_test.split('/')[-1].split('.', 3)[:-1])) + open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            print("Model name:", mname)
             
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses, scores = [], [], []
                print("Iterator:", len(X), hp.batch_size)
                for i in range(len(X) // hp.batch_size):                
                    print('Step:\t', i)     
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    

                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        #print('Inspecting:', source, target, pred)
                        #got = " ".join(idx2zh[idx] for idx in pred).split("。", 2)[0].strip() + ' 。'
                        #got = ''.join(idx2zh[idx] for idx in pred).split('。')[0].strip() 
                        got = ' '.join(idx2zh[idx] for idx in pred).split('</S>')[0]
                        if task_name == 'jieba':
                            fout.write("- source: " + source +"\n")
                            fout.write("- expected: " + ' '.join(cut(source, target)) + "\n")
                            fout.write("- got: " + ' '.join(cut(source, got)) + "\n\n")
                            fout.flush()
                        else:
                            fout.write("- source: " + source +"\n")
                            fout.write("- expected: " + target + "\n")
                            fout.write("- got: " + got + "\n\n")
                            fout.flush()
                        

                        # accumlate accuracty
                        ref = cut(source, target)
                        hypothesis = cut(source, got)
                        acc = len([x for x in hypothesis if x in ref])/len(ref)
                        scores.append(min(1, acc))
 
                                 

                ## Calculate bleu score
                #score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Tokenization Accuracy = " + str(100*(sum(scores)/len(scores))))

                                          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choice the task you want to eval.')
    parser.add_argument('--task', help='task name(default: seq2seq)')

    args = parser.parse_args()
    task_name = args.task
    eval(task_name)
    print("Done")
    

