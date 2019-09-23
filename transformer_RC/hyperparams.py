# -*- coding: utf-8 -*-
#/usr/bin/python3



class rc_Hyperparams:
    trainset = './datasets/train_round_0.csv'
    testset = './datasets/test_data_r0.csv'

    trainfile = './preprocessed/train.csv'
    testfile = './preprocessed/test.csv'


    batch_size = 64 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'rc_model_dir' # log directory
    
    # model
    q_maxlen = 50
    p_maxlen = 200
    ans_maxlen = 40
    min_cnt = 3 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 256 # alias = C
    num_blocks = 5 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.5
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    #n_class = 2
    dropout_keep_prob = 0.55
    reg_lambda = 0.1
    Passage_fuse = 'bi-rnn' # bi-rnn or Pooling
    use_dropout = True
    weight_decay = 0.1





class seq2seq_Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = './datasets/zh-en/train.tags.zh-en.en'
    target_train = './datasets/zh-en/train.tags.zh-en.zh' 
    source_test = './datasets/zh-en/IWSLT15.TED.tst2011.zh-en.en.xml' 
    target_test = './datasets/zh-en/IWSLT15.TED.tst2011.zh-en.zh.xml'

    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'seq2seq_model_dir' # log directory
    
    # model
    maxlen = 100 # Maximum number of words in a sentence. alias = T.
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


class infersent_Block_Hyperparams:
    '''Hyperparameters'''
    # data
    trainset = './opensrc_dta/train.csv'
    testset = './opensrc_dta/test.csv' 


    # training
    relations = {'entailment': '0', 'contradiction': '1', 'neutral': '2'}

    batch_size = 64 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'infersent_model_dir' # log directory
    
    # model
    maxlen = 24 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 3 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 5 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    #n_class = 2
    dropout_keep_prob = 0.55
    reg_lambda = 0.1





    
    
    
