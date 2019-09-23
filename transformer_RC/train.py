# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf

from hyperparams import rc_Hyperparams as hp
from data_load import get_batch_data, load_vocabs
from modules import *
import os, codecs
from tqdm import tqdm
from models import vanilla_transformer

# custom wrapper
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder



class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.q, self.p, self.q_length, self.p_length, \
                self.start_label, self.end_label, self.num_batch = get_batch_data() 
                self.dropout_keep_prob = hp.dropout_keep_prob

            else: # inference
                self.q = tf.placeholder(tf.int32, [None, hp.q_maxlen])
                self.p = tf.placeholder(tf.int32, [None, hp.p_maxlen])
                self.q_length = tf.placeholder(tf.int32, [None])
                self.p_length = tf.placeholder(tf.int32, [None])
                self.start_label = tf.placeholder(tf.int32, [None])
                self.end_label = tf.placeholder(tf.int32, [None])

            self.dropout_keep_prob = hp.dropout_keep_prob
            self.l2_loss = tf.constant(0.0)
            # define decoder input
            self.decoder_inputs = tf.concat((tf.ones_like(self.p[:, :1])*2, self.p[:, :-1]), -1) # 2:<S>

            # Load vocabulary    
            word2idx, idx2word = load_vocabs()

            # initialize transformer
            transformer = vanilla_transformer(hp, self.is_training)
            ### encode
            self.q_encodes, self.p_encodes = transformer.encode(self.q, len(word2idx)), \
                transformer.encode(self.q, len(word2idx))


            #concated features to attend p with q
            # first pad q_encodes to the length of p_encodes
            pad_dim = hp.p_maxlen - hp.q_maxlen
            pad_ = tf.zeros([tf.shape(self.q_encodes)[0], pad_dim, hp.hidden_units], dtype = self.q_encodes.dtype)
            self.padded_q_encodes = tf.concat([self.q_encodes, pad_,], 1)
            #normalization
            self.padded_q_encodes = normalize(self.padded_q_encodes)

            # Decoder
            self.dec = transformer.decode(self.decoder_inputs, self.padded_q_encodes, len(word2idx), hp.p_maxlen)

            # fix paragraph tensor with self.dec
            self.p_encodes = self.dec

            """
            The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
            """
            match_layer = AttentionFlowMatchLayer(hp.hidden_units)
            self.match_p_encodes, _ = match_layer.match(self.p_encodes, self.q_encodes,
                                                        self.p_length, self.q_length)

            # pooling or bi-rnn to fuision passage encodes
            if hp.Passage_fuse == 'Pooling':
                #pooling layer
                self.match_p_encodes = \
                tf.keras.layers.MaxPool1D(pool_size=4, strides=None, padding='valid')\
                                        (self.match_p_encodes)

                self.match_p_encodes = tf.reshape(self.match_p_encodes, [-1, hp.p_maxlen, hp.hidden_units])
                #normalization
                self.match_p_encodes = tf.layers.batch_normalization(self.match_p_encodes)
                if hp.use_dropout:
                    self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)
            elif hp.Passage_fuse == 'bi-rnn':
                self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                             hp.hidden_units, layer_num=1, concat = False)
                if hp.use_dropout:
                    self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)


            decoder = PointerNetDecoder(hp.hidden_units)
            self.start_probs, self.end_probs = decoder.decode(self.match_p_encodes,
                                                              self.q_encodes)

                
            if is_training:  
                self.start_loss = self.sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
                self.end_loss = self.sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
                self.all_params = tf.trainable_variables()
                self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
                if hp.weight_decay > 0:
                    with tf.variable_scope('l2_loss'):
                        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                    self.loss += hp.weight_decay * l2_loss



                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.loss)
                self.merged = tf.summary.merge_all()



    def sparse_nll_loss(self, probs, labels, epsilon=1e-9, scope=None):
        """
        negative log likelyhood loss
        """
        with tf.name_scope(scope, "log_loss"):
            labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
            losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
        return losses


if __name__ == '__main__':                
    # Load vocabulary    
    word2idx, idx2word = load_vocabs()
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
      with open('acc_mean_loss.rec', 'w') as rec:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                #acc, los = sess.run(g.acc), sess.run(g.mean_loss)
                los = sess.run(g.loss)
                if not los > float('-inf'):
                  print("loss: ",los)
                  gs = sess.run(g.global_step) 
                  sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
                  break
   
                rec.write('epochs {}\tstep {}\t{}\t{}\n'.format(epoch, step, 'Loss:', los))

            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    
    print("Done")
    

