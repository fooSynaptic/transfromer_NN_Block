# -*- coding: utf-8 -*-
"""
Hyperparameter configurations for different NLP tasks.

This module contains configuration classes for:
- Reading Comprehension (RC)
- Sequence-to-Sequence (NMT)
- Text Classification
- Natural Language Inference (InferSent)

Example:
    >>> from hyperparams import RCHyperparams, Seq2SeqHyperparams
    >>> rc_config = RCHyperparams()
    >>> nmt_config = Seq2SeqHyperparams()
"""


class BaseHyperparams:
    """
    Base hyperparameters shared across all tasks.
    
    Attributes:
        batch_size: Training batch size
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        hidden_units: Model hidden dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        min_cnt: Minimum word frequency for vocabulary
        sinusoid: Use sinusoidal positional encoding (vs learned)
    """
    
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
    """
    Hyperparameters for Reading Comprehension task.
    
    Based on BiDAF + Transformer architecture for extractive QA.
    
    Attributes:
        question_maxlen: Maximum question length
        passage_maxlen: Maximum passage length
        answer_maxlen: Maximum answer length
        dropout_keep_prob: Keep probability for dropout
        reg_lambda: L2 regularization coefficient
    """
    
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
    """
    Hyperparameters for Sequence-to-Sequence (NMT) task.
    
    Used for English-to-Chinese neural machine translation.
    Dataset: WIT3 (Web Inventory of Transcribed and Translated Talks)
    """
    
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
    """
    Hyperparameters for Text Classification task.
    
    Used for Chinese text classification (THU-CTC dataset).
    Categories: 时尚, 教育, 时政, 体育, 游戏, 家居, 科技, 房产, 财经, 娱乐
    """
    
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
    """
    Hyperparameters for Natural Language Inference task.
    
    Based on Stanford SNLI dataset.
    Labels: entailment, contradiction, neutral
    
    Attributes:
        relations: Mapping from label names to class indices
    """
    
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
