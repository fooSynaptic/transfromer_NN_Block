# -*- coding: utf-8 -*-
#/usr/bin/python3


class seq2seq_Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = './dataset/textSummary/train.tags.src-tgt.src'
    target_train = './dataset/textSummary/train.tags.tgt-src.tgt' 
    source_test = './dataset/textSummary/test.tags.src-tgt.src' 
    target_test = './dataset/textSummary/test.tags.tgt-src.tgt'

    # training
    batch_size = 4 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'seq2seq_model_dir' # log directory
    
    # model
    maxlen = 500 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 3 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 5 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.




class feature_Block_Hyperparams:
    '''Hyperparameters'''
    # data
    trainset = './datasets/trainset.txt'
    testset = './datasets/testset.txt' 


    # training
    batch_size = 4 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'Block_model_dir' # log directory
    
    # model
    maxlen = 500 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 3 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 5 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    n_class = 2







    
    
    
