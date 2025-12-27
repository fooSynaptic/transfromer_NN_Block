# -*- coding: utf-8 -*-
"""
Transformer Model Implementation.

This module provides a vanilla Transformer implementation following the paper
"Attention Is All You Need" (Vaswani et al., 2017).

Classes:
    VanillaTransformer: Standard encoder-decoder Transformer architecture.
"""

import tensorflow as tf
from modules import (
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
        hyperparams: Configuration object containing model hyperparameters.
        is_training: Boolean indicating training mode (affects dropout).
    
    Attributes:
        hp: Hyperparameters configuration.
        is_training: Training mode flag.
    """
    
    def __init__(self, hyperparams, is_training):
        self.hp = hyperparams
        self.is_training = is_training

    def encode(self, inputs, vocab_size):
        """
        Encode input sequence using Transformer encoder.
        
        The encoder consists of:
        1. Token embedding with optional scaling
        2. Positional encoding (sinusoidal or learned)
        3. Dropout layer
        4. N stacked encoder blocks (self-attention + feed-forward)
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_length].
            vocab_size: Size of the vocabulary.
            
        Returns:
            Encoded representations of shape [batch_size, seq_length, hidden_units].
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # Token embedding
            enc = embedding(
                inputs,
                vocab_size=vocab_size,
                num_units=self.hp.hidden_units,
                scale=True,
                scope="enc_embed"
            )
            
            # Positional encoding
            if self.hp.sinusoid:
                enc += positional_encoding(
                    inputs,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe"
                )
            else:
                # Learned positional embeddings
                position_ids = tf.tile(
                    tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0),
                    [tf.shape(inputs)[0], 1]
                )
                enc += embedding(
                    position_ids,
                    vocab_size=vocab_size,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe"
                )

            # Dropout
            enc = tf.layers.dropout(
                enc,
                rate=self.hp.dropout_rate,
                training=tf.convert_to_tensor(self.is_training)
            )
            
            # Stacked encoder blocks
            for block_idx in range(self.hp.num_blocks):
                with tf.variable_scope(f"encoder_block_{block_idx}", reuse=tf.AUTO_REUSE):
                    # Multi-head self-attention
                    enc = multihead_attention(
                        queries=enc,
                        keys=enc,
                        num_units=self.hp.hidden_units,
                        num_heads=self.hp.num_heads,
                        dropout_rate=self.hp.dropout_rate,
                        is_training=self.is_training,
                        causality=False
                    )
                    
                    # Position-wise feed-forward network
                    enc = feedforward(
                        enc,
                        num_units=[4 * self.hp.hidden_units, self.hp.hidden_units]
                    )
        
        return enc

    def decode(self, decoder_inputs, encoder_outputs, vocab_size, max_length):
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
            decoder_inputs: Decoder input tensor of shape [batch_size, seq_length].
            encoder_outputs: Encoder output tensor for cross-attention.
            vocab_size: Size of the vocabulary.
            max_length: Maximum sequence length for positional encoding.
            
        Returns:
            Decoded representations of shape [batch_size, seq_length, hidden_units].
        """
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # Token embedding
            dec = embedding(
                decoder_inputs,
                vocab_size=vocab_size,
                num_units=self.hp.hidden_units,
                scale=True,
                scope="dec_embed"
            )
            
            # Positional encoding
            if self.hp.sinusoid:
                dec += positional_encoding(
                    decoder_inputs,
                    vocab_size=max_length,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe"
                )
            else:
                # Learned positional embeddings
                position_ids = tf.tile(
                    tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0),
                    [tf.shape(decoder_inputs)[0], 1]
                )
                dec += embedding(
                    position_ids,
                    vocab_size=max_length,
                    num_units=self.hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe"
                )
            
            # Dropout
            dec = tf.layers.dropout(
                dec,
                rate=self.hp.dropout_rate,
                training=tf.convert_to_tensor(self.is_training)
            )
            
            # Stacked decoder blocks
            for block_idx in range(self.hp.num_blocks):
                with tf.variable_scope(f"decoder_block_{block_idx}"):
                    # Masked self-attention (causal)
                    dec = multihead_attention(
                        queries=dec,
                        keys=dec,
                        num_units=self.hp.hidden_units,
                        num_heads=self.hp.num_heads,
                        dropout_rate=self.hp.dropout_rate,
                        is_training=self.is_training,
                        causality=True,
                        scope="self_attention"
                    )

                    # Cross-attention to encoder outputs
                    dec = multihead_attention(
                        queries=dec,
                        keys=encoder_outputs,
                        num_units=self.hp.hidden_units,
                        num_heads=self.hp.num_heads,
                        dropout_rate=self.hp.dropout_rate,
                        is_training=self.is_training,
                        causality=False,
                        scope="cross_attention"
                    )

                    # Position-wise feed-forward network
                    dec = feedforward(
                        dec,
                        num_units=[4 * self.hp.hidden_units, self.hp.hidden_units]
                    )
        
        return dec


# Backward compatibility alias
vanilla_transformer = VanillaTransformer
