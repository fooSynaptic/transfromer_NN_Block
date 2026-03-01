# -*- coding: utf-8 -*-
"""
Transformer Model Implementation.

This module provides a vanilla Transformer implementation following the paper
"Attention Is All You Need" (Vaswani et al., 2017).

Classes:
    VanillaTransformer: Standard encoder-decoder Transformer architecture.

Example:
    >>> from models import VanillaTransformer
    >>> from hyperparams import Seq2SeqHyperparams
    >>> hp = Seq2SeqHyperparams()
    >>> model = VanillaTransformer(hp, is_training=True)
    >>> enc_output = model.encode(inputs, vocab_size=10000)
"""

import tensorflow as tf

# Import modules from sibling directories
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules import (
        embedding,
        positional_encoding,
        multihead_attention,
        feedforward,
        normalize
    )
except ImportError:
    from en_zh_NMT.modules import (
        embedding,
        positional_encoding,
        multihead_attention,
        feedforward,
        normalize
    )


class VanillaTransformer:
    """
    Vanilla Transformer model with encoder-decoder architecture.
    
    This implementation follows the original Transformer paper with:
    - Multi-head self-attention layers
    - Position-wise feed-forward networks
    - Residual connections and layer normalization
    - Positional encoding (sinusoidal or learned)
    
    Args:
        hyperparams: Configuration object containing model hyperparameters
        is_training: Boolean indicating training mode (affects dropout)
    
    Attributes:
        hp: Hyperparameters configuration
        is_training: Training mode flag
    """
    
    def __init__(self, hyperparams, is_training: bool):
        """
        Initialize the Transformer model.
        
        Args:
            hyperparams: Hyperparameter configuration object
            is_training: Whether in training mode
        """
        self.hp = hyperparams
        self.is_training = is_training

    def encode(self, inputs: tf.Tensor, vocab_size: int) -> tf.Tensor:
        """
        Encode input sequence using Transformer encoder.
        
        The encoder consists of:
        1. Token embedding with optional scaling
        2. Positional encoding (sinusoidal or learned)
        3. Dropout layer
        4. N stacked encoder blocks (self-attention + feed-forward)
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_length]
            vocab_size: Size of the vocabulary
            
        Returns:
            Encoded representations of shape [batch_size, seq_length, hidden_units]
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # Token embedding
            encoded = embedding(
                inputs,
                vocab_size=vocab_size,
                num_units=self.hp.hidden_units,
                scale=True,
                scope="enc_embed"
            )
            
            # Positional encoding
            if self.hp.sinusoid:
                encoded += positional_encoding(
                    inputs,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe"
                )
            else:
                # Learned positional embeddings
                seq_length = tf.shape(inputs)[1]
                position_ids = tf.tile(
                    tf.expand_dims(tf.range(seq_length), 0),
                    [tf.shape(inputs)[0], 1]
                )
                encoded += embedding(
                    position_ids,
                    vocab_size=vocab_size,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe"
                )

            # Dropout
            encoded = tf.layers.dropout(
                encoded,
                rate=self.hp.dropout_rate,
                training=tf.convert_to_tensor(self.is_training)
            )
            
            # Stacked encoder blocks
            for block_idx in range(self.hp.num_blocks):
                with tf.variable_scope(f"encoder_block_{block_idx}", reuse=tf.AUTO_REUSE):
                    # Multi-head self-attention
                    encoded = multihead_attention(
                        queries=encoded,
                        keys=encoded,
                        num_units=self.hp.hidden_units,
                        num_heads=self.hp.num_heads,
                        dropout_rate=self.hp.dropout_rate,
                        is_training=self.is_training,
                        causality=False
                    )
                    
                    # Position-wise feed-forward network
                    encoded = feedforward(
                        encoded,
                        num_units=[4 * self.hp.hidden_units, self.hp.hidden_units]
                    )
        
        return encoded

    def decode(self, decoder_inputs: tf.Tensor, encoder_outputs: tf.Tensor, 
               vocab_size: int, max_length: int) -> tf.Tensor:
        """
        Decode using Transformer decoder with cross-attention.
        
        The decoder consists of:
        1. Token embedding with optional scaling
        2. Positional encoding (sinusoidal or learned)
        3. Dropout layer
        4. N stacked decoder blocks:
           - Masked self-attention (causal)
           - Cross-attention to encoder outputs
           - Feed-forward network
        
        Args:
            decoder_inputs: Decoder input tensor [batch_size, seq_length]
            encoder_outputs: Encoder output tensor for cross-attention
            vocab_size: Size of the vocabulary
            max_length: Maximum sequence length for positional encoding
            
        Returns:
            Decoded representations [batch_size, seq_length, hidden_units]
        """
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # Token embedding
            decoded = embedding(
                decoder_inputs,
                vocab_size=vocab_size,
                num_units=self.hp.hidden_units,
                scale=True,
                scope="dec_embed"
            )
            
            # Positional encoding
            if self.hp.sinusoid:
                decoded += positional_encoding(
                    decoder_inputs,
                    vocab_size=max_length,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe"
                )
            else:
                # Learned positional embeddings
                seq_length = tf.shape(decoder_inputs)[1]
                position_ids = tf.tile(
                    tf.expand_dims(tf.range(seq_length), 0),
                    [tf.shape(decoder_inputs)[0], 1]
                )
                decoded += embedding(
                    position_ids,
                    vocab_size=max_length,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe"
                )
            
            # Dropout
            decoded = tf.layers.dropout(
                decoded,
                rate=self.hp.dropout_rate,
                training=tf.convert_to_tensor(self.is_training)
            )
            
            # Stacked decoder blocks
            for block_idx in range(self.hp.num_blocks):
                with tf.variable_scope(f"decoder_block_{block_idx}"):
                    # Masked self-attention (causal)
                    decoded = multihead_attention(
                        queries=decoded,
                        keys=decoded,
                        num_units=self.hp.hidden_units,
                        num_heads=self.hp.num_heads,
                        dropout_rate=self.hp.dropout_rate,
                        is_training=self.is_training,
                        causality=True,
                        scope="self_attention"
                    )

                    # Cross-attention to encoder outputs
                    decoded = multihead_attention(
                        queries=decoded,
                        keys=encoder_outputs,
                        num_units=self.hp.hidden_units,
                        num_heads=self.hp.num_heads,
                        dropout_rate=self.hp.dropout_rate,
                        is_training=self.is_training,
                        causality=False,
                        scope="cross_attention"
                    )

                    # Position-wise feed-forward network
                    decoded = feedforward(
                        decoded,
                        num_units=[4 * self.hp.hidden_units, self.hp.hidden_units]
                    )
        
        return decoded


# Backward compatibility alias
vanilla_transformer = VanillaTransformer
