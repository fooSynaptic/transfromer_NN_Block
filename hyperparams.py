# -*- coding: utf-8 -*-
"""
Hyperparameter configurations for different NLP tasks.

This module contains configuration classes for:
- Reading Comprehension (RC)
- Sequence-to-Sequence (NMT)
- Text Classification
- Natural Language Inference (InferSent)
"""


class BaseHyperparams:
    """Base hyperparameters shared across all tasks."""
    
    # Training
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 20
    
    # Model architecture
    hidden_units = 512
    num_blocks = 5
    num_heads = 8
    dropout_rate = 0.1
    
    # Vocabulary
    min_cnt = 3  # Words with frequency < min_cnt are encoded as <UNK>
    
    # Positional encoding
    sinusoid = False  # If True, use sinusoidal PE; otherwise, learned embeddings


class RCHyperparams(BaseHyperparams):
    """Hyperparameters for Reading Comprehension task."""
    
    # Data paths
    trainset = './datasets/train_round_0.csv'
    testset = './datasets/test_data_r0.csv'
    trainfile = './preprocessed/train.csv'
    testfile = './preprocessed/test.csv'
    
    # Training
    batch_size = 64
    learning_rate = 0.0001
    logdir = 'rc_model_dir'
    
    # Model
    question_maxlen = 50
    passage_maxlen = 200
    answer_maxlen = 40
    hidden_units = 256
    
    # Regularization
    dropout_rate = 0.5
    dropout_keep_prob = 0.55
    reg_lambda = 0.1
    use_dropout = True
    weight_decay = 0.1


class Seq2SeqHyperparams(BaseHyperparams):
    """Hyperparameters for Sequence-to-Sequence (NMT) task."""
    
    # Data paths
    source_train = './datasets/zh-en/train.tags.zh-en.en'
    target_train = './datasets/zh-en/train.tags.zh-en.zh'
    source_test = './datasets/zh-en/IWSLT15.TED.tst2011.zh-en.en.xml'
    target_test = './datasets/zh-en/IWSLT15.TED.tst2011.zh-en.zh.xml'
    
    # Training
    batch_size = 32
    logdir = 'seq2seq_model_dir'
    
    # Model
    maxlen = 100  # Maximum sequence length


class TextClassificationHyperparams(BaseHyperparams):
    """Hyperparameters for Text Classification task."""
    
    # Data paths
    trainset = './datasets/trainset.txt'
    testset = './datasets/testset.txt'
    
    # Training
    batch_size = 4
    logdir = 'Block_model_dir'
    
    # Model
    maxlen = 500  # Maximum sequence length
    num_classes = 2


class InferSentHyperparams(BaseHyperparams):
    """Hyperparameters for Natural Language Inference task."""
    
    # Data paths
    trainset = './opensrc_dta/train.csv'
    testset = './opensrc_dta/test.csv'
    
    # Label mapping
    relations = {
        'entailment': '0',
        'contradiction': '1',
        'neutral': '2'
    }
    
    # Training
    batch_size = 64
    logdir = 'infersent_model_dir'
    
    # Model
    maxlen = 24  # Maximum sequence length
    
    # Regularization
    dropout_keep_prob = 0.55
    reg_lambda = 0.1


# Backward compatibility aliases
rc_Hyperparams = RCHyperparams
seq2seq_Hyperparams = Seq2SeqHyperparams
feature_Block_Hyperparams = TextClassificationHyperparams
infersent_Block_Hyperparams = InferSentHyperparams
