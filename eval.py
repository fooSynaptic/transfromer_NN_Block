# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_de_vocab, load_en_vocab, load_train_data, create_data
from train import Graph
from nltk.translate.bleu_score import corpus_bleu


def load_test_data(test_source, test_target):
    def _refine(line):
        #line = re.sub("<[^>]+>", "", line)
        #line = re.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(test_source, 'r', 'utf-8').read().split("\n") if line and line[:4] != "<seg"]
    en_sents = [_refine(line) for line in codecs.open(test_target, 'r', 'utf-8').read().split("\n") if line and line[:4] != "<seg"]
        
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)


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






def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Sources, Targets = load_test_data(hp.source_train, hp.target_train)
    print(X, Sources, Targets)
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
     
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
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
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    

                    print('Results:',sources, targets, preds) 
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        #print('Inspecting:', source, target, pred)
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        print('Inspecting:', source, target, got, len(source), len(target), len(got))
                        if len(got) < len(target): got += target[len(got):]
                        fout.write("- expected: " + cut(source, target) + "\n")
                        #except:
                        #    fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + cut(source, got) + "\n\n")
                        fout.flush()
                        
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
                                 

                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
