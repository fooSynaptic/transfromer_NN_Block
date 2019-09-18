# -*- coding: utf-8 -*-
#/usr/bin/python3

import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import rc_Hyperparams as hp
from data_load import load_vocabs, load_train_data, load_test_data, create_data
from train import Graph
#from nltk.translate.bleu_score import corpus_bleu
import argparse
#from sklearn.metrics import classification_report
#from utils import compute_bleu_rouge
import pandas as pd
from modules import bleu


def find_best_answer_for_passage(start_probs, end_probs, passage_len=None):
    """
    Finds the best answer with the maximum start_prob * end_prob from a single passage
    """
    if passage_len is None:
        passage_len = len(start_probs)
    else:
        passage_len = min(len(start_probs), passage_len)

    best_start, best_end, max_prob = -1, -1, 0

    for start_idx in range(passage_len):
        #within the span of answer limit
        for ans_len in range(hp.ans_maxlen):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue

            prob = start_probs[start_idx] * end_probs[end_idx]
            if prob > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob
    return (best_start, best_end), max_prob





 

def eval(task_name):
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    test_data = pd.read_csv(hp.testfile)
    questions, contents, q_lens, p_lens, start_pos, end_pos = load_test_data()
    raw_passages = list(test_data['content'])
    reference_answers = list(test_data['answer'])


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
                
                pred_answers, ref_answers = [], []
                pred_dict, ref_dict = {}, {}
                ques_id = 0
                eval_dict = {'bleu_1':[], 'bleu_2':[], 'bleu_3':[], 'bleu_4':[]}

                for i in range(len(questions) // hp.batch_size):                
                    print("Iterator: {} / {}".format(i, len(questions)//hp.batch_size))    

                    ### Get mini-batches
                    q = questions[i*hp.batch_size: (i+1)*hp.batch_size]
                    p = contents[i*hp.batch_size: (i+1)*hp.batch_size]
                    q_length = q_lens[i*hp.batch_size: (i+1)*hp.batch_size]
                    p_length = p_lens[i*hp.batch_size: (i+1)*hp.batch_size]
                    s_pos = start_pos[i*hp.batch_size: (i+1)*hp.batch_size]
                    e_pos = end_pos[i*hp.batch_size: (i+1)*hp.batch_size]
                    passages = raw_passages[i*hp.batch_size: (i+1)*hp.batch_size]
                    ref_answers = reference_answers[i*hp.batch_size: (i+1)*hp.batch_size]

                    feed_dict = {g.q: q,
                                g.p: p,
                                g.q_length: q_length,
                                g.p_length: p_length,
                                g.start_label: s_pos,
                                g.end_label: e_pos}

                    start_probs, end_probs = sess.run([g.start_probs, g.end_probs], feed_dict)


                    ### Write to file
                    for start_prob, end_prob, passage, ref in zip(start_probs, end_probs, passages, ref_answers):
                        pred_span, prob = find_best_answer_for_passage(start_prob, end_prob)
                        pred_answer = passage[pred_span[0]: pred_span[1]+1]
                        
                        if not len(pred_answer) > 0: continue

                        pred_dict[str(ques_id)] = [pred_answer]
                        ref_dict[str(ques_id)] = [ref]
                        ques_id += 1

                        fout.write('-ref: '+ ref)
                        fout.write("-pred: "+ pred_answer)

                        b1, b2, b3, b4 = bleu(list(pred_answer), list(ref), 1), \
                                         bleu(list(pred_answer), list(ref), 2), \
                                         bleu(list(pred_answer), list(ref), 3), \
                                         bleu(list(pred_answer), list(ref), 4)
 

                        eval_dict['bleu_1'].append(b1)
                        eval_dict['bleu_2'].append(b2)
                        eval_dict['bleu_3'].append(b3)
                        eval_dict['bleu_2'].append(b2)
                
                for metric in eval_dict:
                    fout.write(metric + '\t' + str(np.mean(eval_dict[metric])) + '\n') 
                    print(metric + '\t' + str(np.mean(eval_dict[metric]))) 
                                          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choice the task you want to eval.')
    parser.add_argument('--task', help='task name(default: RC)')

    args = parser.parse_args()
    task_name = args.task
    eval(task_name)
    print("Done")
    

