# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    if False:
        source_train = './corpora/train.tags.de-en.de'
        target_train = './corpora/train.tags.de-en.en'
        source_test = './corpora/IWSLT16.TED.tst2014.de-en.de.xml'
        target_test = './corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    target_train = './dataset/train.tags.en-zh.en'   
    source_train = './dataset/train.tags.zh-en.zh'
    target_test = './dataset/test.tags.en-zh.en'
    source_test = './dataset/test.tags.zh-en.zh' 


    # training
    batch_size = 8 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'jieba_dir' # log directory
    
    # model
    maxlen = 86 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 3 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
    
    
