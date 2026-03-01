# -*- coding: utf-8 -*-
"""
Transformer Building Blocks.

This module implements core components of the Transformer architecture:
- Layer normalization
- Positional encoding
- Multi-head attention
- Position-wise feed-forward networks
- Label smoothing
- BLEU score calculation

Reference:
    "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762
"""

import numpy as np
import collections
import math
import tensorflow as tf


def normalize(inputs: tf.Tensor, epsilon: float = 1e-8,
              scope: str = "ln", reuse: bool = None) -> tf.Tensor:
    """
    Applies layer normalization.
    
    Layer normalization normalizes the inputs across the features instead of
    the batch dimension, making it suitable for sequential data.
    
    Args:
        inputs: A tensor with 2 or more dimensions
        epsilon: Small constant for numerical stability
        scope: Variable scope name
        reuse: Whether to reuse variables
        
    Returns:
        Normalized tensor with same shape as inputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        params_shape = inputs.get_shape()[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
        
    return outputs


def embedding(inputs: tf.Tensor, vocab_size: int, num_units: int,
              zero_pad: bool = True, scale: bool = True,
              scope: str = "embedding", reuse: bool = None) -> tf.Tensor:
    """
    Embedding lookup layer.
    
    Args:
        inputs: Tensor of token indices
        vocab_size: Vocabulary size
        num_units: Embedding dimension
        zero_pad: If True, first row (id 0) is zeros (for padding)
        scale: If True, scale embeddings by sqrt(num_units)
        scope: Variable scope name
        reuse: Whether to reuse variables
        
    Returns:
        Embedded tensor with shape [..., num_units]
    """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable(
            'lookup_table',
            dtype=tf.float32,
            shape=[vocab_size, num_units],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        if zero_pad:
            lookup_table = tf.concat(
                (tf.zeros(shape=[1, num_units]), lookup_table[1:, :]),
                axis=0
            )
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5)
            
    return outputs


def positional_encoding(inputs: tf.Tensor, num_units: int,
                        zero_pad: bool = True, scale: bool = True,
                        scope: str = "positional_encoding",
                        reuse: bool = None) -> tf.Tensor:
    """
    Sinusoidal positional encoding.
    
    Uses sine and cosine functions of different frequencies to encode
    position information.
    
    Args:
        inputs: Input tensor for shape reference
        num_units: Encoding dimension
        zero_pad: If True, pad position 0 with zeros
        scale: If True, scale by sqrt(num_units)
        scope: Variable scope name
        reuse: Whether to reuse variables
        
    Returns:
        Positional encoding tensor
    """
    batch_size, seq_length = inputs.get_shape().as_list()
    
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])

        # Create position encodings
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(seq_length)
        ])

        # Apply sin to even indices, cos to odd indices
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat(
                (tf.zeros(shape=[1, num_units]), lookup_table[1:, :]),
                axis=0
            )
            
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * (num_units ** 0.5)

        return outputs


def multihead_attention(queries: tf.Tensor, keys: tf.Tensor,
                        num_units: int = None, num_heads: int = 8,
                        dropout_rate: float = 0, is_training: bool = True,
                        causality: bool = False,
                        scope: str = "multihead_attention",
                        reuse: bool = None) -> tf.Tensor:
    """
    Multi-head attention mechanism.
    
    Projects queries, keys, and values into multiple subspaces,
    performs attention in parallel, and concatenates results.
    
    Args:
        queries: Query tensor [batch_size, query_len, depth]
        keys: Key tensor [batch_size, key_len, depth]
        num_units: Attention dimension (defaults to query depth)
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        is_training: Training mode flag
        causality: If True, mask future positions (for decoder)
        scope: Variable scope name
        reuse: Whether to reuse variables
        
    Returns:
        Attention output [batch_size, query_len, num_units]
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set default attention size
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        
        # Split and concat for multi-head
        Q_split = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_split = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_split = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Attention scores
        outputs = tf.matmul(Q_split, tf.transpose(K_split, [0, 2, 1]))
        outputs = outputs / (K_split.get_shape().as_list()[-1] ** 0.5)
        
        # Key masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), 
                           [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
  
        # Causality mask (for decoder self-attention)
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), 
                          [tf.shape(outputs)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)
  
        # Softmax
        outputs = tf.nn.softmax(outputs)
         
        # Query masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), 
                             [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks
          
        # Dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate,
                                    training=tf.convert_to_tensor(is_training))
               
        # Weighted sum of values
        outputs = tf.matmul(outputs, V_split)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
              
        # Residual and normalization
        outputs += queries
        outputs = normalize(outputs)
 
    return outputs


def feedforward(inputs: tf.Tensor, num_units: List[int] = [2048, 512],
                scope: str = "feedforward", reuse: bool = None) -> tf.Tensor:
    """
    Position-wise feed-forward network.
    
    Applies two linear transformations with ReLU activation:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        inputs: Input tensor [batch_size, seq_len, depth]
        num_units: List of [inner_dim, output_dim]
        scope: Variable scope name
        reuse: Whether to reuse variables
        
    Returns:
        Output tensor with same shape as inputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        inner_params = {
            "inputs": inputs,
            "filters": num_units[0],
            "kernel_size": 1,
            "activation": tf.nn.relu,
            "use_bias": True
        }
        inner_output = tf.layers.conv1d(**inner_params)
        
        # Readout layer
        output_params = {
            "inputs": inner_output,
            "filters": num_units[1],
            "kernel_size": 1,
            "activation": None,
            "use_bias": True
        }
        outputs = tf.layers.conv1d(**output_params)
        
        # Residual and normalization
        outputs += inputs
        outputs = normalize(outputs)
    
    return outputs


def label_smoothing(inputs: tf.Tensor, epsilon: float = 0.1) -> tf.Tensor:
    """
    Applies label smoothing.
    
    Replaces hard 0/1 labels with soft labels:
    soft_label = (1 - epsilon) * hard_label + epsilon / num_classes
    
    Args:
        inputs: One-hot label tensor
        epsilon: Smoothing factor
        
    Returns:
        Smoothed label tensor
        
    Reference:
        https://arxiv.org/abs/1512.00567
    """
    num_classes = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / num_classes)


def bleu_score(pred_tokens: List[str], label_tokens: List[str], 
               max_n: int = 4) -> float:
    """
    Calculate BLEU score for machine translation evaluation.
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram precision
    with brevity penalty.
    
    Args:
        pred_tokens: Predicted token sequence
        label_tokens: Reference token sequence
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score between 0 and 1
    """
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    
    for n in range(1, max_n + 1):
        num_matches = 0
        label_subs = collections.defaultdict(int)
        
        # Count n-grams in reference
        for i in range(len_label - n + 1):
            ngram = ''.join(label_tokens[i: i + n])
            label_subs[ngram] += 1
        
        # Count matching n-grams in prediction
        for i in range(len_pred - n + 1):
            ngram = ''.join(pred_tokens[i: i + n])
            if label_subs[ngram] > 0:
                num_matches += 1
                label_subs[ngram] -= 1
        
        # Update score with precision
        denominator = len_pred - n + 1
        if denominator > 0:
            precision = num_matches / denominator
            score *= math.pow(precision, math.pow(0.5, n))
    
    return score
