# encoding = utf-8
# /usr/bin/python3
import tensorflow as tf
from modules import *


class vanilla_transformer():
    def __init__(self, hp, is_training):
        self.hp = hp
        self.train = is_training

    def encode(self, Input, Vocabs_length):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            ## Embedding
            enc = embedding(Input,
                                vocab_size=Vocabs_length, 
                                num_units=self.hp.hidden_units, 
                                scale=True,
                                scope="enc_embed")
            
            ## Positional Encoding
            if self.hp.sinusoid:
                enc += positional_encoding(Input,
                                num_units=self.hp.hidden_units, 
                                zero_pad=False, 
                                scale=False,
                                scope="enc_pe")
            else:
                enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(Input)[1]), 0), [tf.shape(Input)[0], 1]),
                                vocab_size=Vocabs_length, 
                                num_units=self.hp.hidden_units, 
                                zero_pad=False, 
                                scale=False,
                                scope="enc_pe")

            ## Dropout
            enc = tf.layers.dropout(enc, 
                                        rate=self.hp.dropout_rate, 
                                        training=tf.convert_to_tensor(self.train))
            
            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks", reuse = tf.AUTO_REUSE):
                    ### Multihead Attention
                    enc = multihead_attention(queries=enc, 
                                                    keys=enc, 
                                                    num_units=self.hp.hidden_units, 
                                                    num_heads=self.hp.num_heads, 
                                                    dropout_rate=self.hp.dropout_rate,
                                                    is_training=self.train,
                                                    causality=False)
                    
                    ### Feed Forward
                    enc = feedforward(enc, num_units=[4*self.hp.hidden_units, self.hp.hidden_units])
        state = enc
        return state

    
    def decode(self, decoder_inputs, key_states, Vocabs_length, decode_length):
        with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
            ## Embedding
            self.dec = embedding(decoder_inputs, 
                                    vocab_size=Vocabs_length, 
                                    num_units=self.hp.hidden_units,
                                    scale=True, 
                                    scope="dec_embed")
            
            ## Positional Encoding
            if self.hp.sinusoid:
                self.dec += positional_encoding(decoder_inputs,
                                    vocab_size=decode_length, 
                                    num_units=self.hp.hidden_units, 
                                    zero_pad=False, 
                                    scale=False,
                                    scope="dec_pe")
            else:
                self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0), [tf.shape(decoder_inputs)[0], 1]),
                                    vocab_size=decode_length, 
                                    num_units=self.hp.hidden_units, 
                                    zero_pad=False,
                                    scale=False,
                                    scope="dec_pe")
            
            ## Dropout
            self.dec = tf.layers.dropout(self.dec, 
                                        rate=self.hp.dropout_rate, 
                                        training=tf.convert_to_tensor(self.train))
            
            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec, 
                                                    keys=self.dec, 
                                                    num_units=self.hp.hidden_units, 
                                                    num_heads=self.hp.num_heads, 
                                                    dropout_rate=self.hp.dropout_rate,
                                                    is_training=self.train,
                                                    causality=True, 
                                                    scope="self_attention")
                    

                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec, 
                                                    keys=key_states,
                                                    num_units=self.hp.hidden_units, 
                                                    num_heads=self.hp.num_heads,
                                                    dropout_rate=self.hp.dropout_rate,
                                                    is_training=self.train, 
                                                    causality=False,
                                                    scope="vanilla_attention")
                    

                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4*self.hp.hidden_units, self.hp.hidden_units])
        
        output_state = self.dec
        return output_state
