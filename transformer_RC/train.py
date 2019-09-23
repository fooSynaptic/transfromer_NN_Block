# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf

from hyperparams import rc_Hyperparams as hp
from data_load import get_batch_data, load_vocabs
from modules import *
import os, codecs
from tqdm import tqdm

#import keras


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
            # define decoder inputs
            #for sentence relationship learning task we want to encoder sent1 to e1, then decoder(e1 + sent2)
            #to get a more sementic relationship across corpus

            # we want enhance the feature of query 
            # so the feature of query is boostted by add feature input into decoder
            # the decoder input should be the same shape of paragraph
            pad_dim = hp.p_maxlen - hp.q_maxlen
            pad_ = tf.zeros([tf.shape(self.q)[0], pad_dim], dtype = self.q.dtype)
            self.decoder_inputs = tf.concat((pad_, tf.ones_like(self.q[:, :1])*2, self.q[:, :-1]), -1) # 2:<S>


            # Load vocabulary    
            word2idx, idx2word = load_vocabs()

            
            # concated Encoder features
            features, concated_features = [self.q, self.p], []
            vocabSizes = [hp.q_maxlen, hp.p_maxlen]

            for i in range(len(features)):
              tmp_feature = features[i]
              with tf.variable_scope("encoder_{}".format(i)):
                  ## Embedding
                  self.enc = embedding(tmp_feature, 
                                        vocab_size=len(word2idx), 
                                        num_units=hp.hidden_units, 
                                        scale=True,
                                        scope="enc_embed")
                  
                  ## Positional Encoding
                  if hp.sinusoid:
                      self.enc += positional_encoding(tmp_feature,
                                        num_units=hp.hidden_units, 
                                        zero_pad=False, 
                                        scale=False,
                                        scope="enc_pe")
                  else:
                      self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(tmp_feature)[1]), 0), [tf.shape(tmp_feature)[0], 1]),
                                        vocab_size=vocabSizes[i], 
                                        num_units=hp.hidden_units, 
                                        zero_pad=False, 
                                        scale=False,
                                        scope="enc_pe")

                  ## Dropout
                  self.enc = tf.layers.dropout(self.enc, 
                                              rate=hp.dropout_rate, 
                                              training=tf.convert_to_tensor(is_training))

                  #store
                  concated_features.append(self.enc)

            #
            self.q_encodes, self.p_encodes = concated_features

            #concated features
            # first pad q_encodes to the length of p_encodes
            pad_dim = hp.p_maxlen - hp.q_maxlen
            pad_ = tf.zeros([tf.shape(self.q_encodes)[0], pad_dim, hp.hidden_units], dtype = self.q_encodes.dtype)

            self.p_fixed_q_encodes = tf.concat([pad_, self.q_encodes], 1)


            self.p_q_encodes = tf.divide(tf.add(self.p_fixed_q_encodes, self.p_encodes), len(concated_features))
            #normalization
            self.p_q_encodes = normalize(self.p_q_encodes)


            ## Blocks
            # enhance concated feature with raw query input
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.p_q_encodes = multihead_attention(queries=self.p_q_encodes, 
                                                    keys=self.p_q_encodes, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False)
                    
                    ### Feed Forward
                    self.p_q_encodes = feedforward(self.p_q_encodes, num_units=[4*hp.hidden_units, hp.hidden_units])
        
        
            #for sentence relationship learning task we want to encoder sent1 to e1, then decoder(e1 + sent2)
            #to get a more sementic relationship across corpus




            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(self.decoder_inputs, 
                                      vocab_size=len(word2idx), 
                                      num_units=hp.hidden_units,
                                      scale=True, 
                                      scope="dec_embed")
                
                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.p_maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        

                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.p_q_encodes, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
            

            # fix paragraph tensor with self.dec
            #self.p_encodes = tf.add(self.p_encodes, tf.multiply(self.p_encodes, self.dec))
            
            self.p_encodes = self.dec




            
            """
            The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
            """
            
            match_layer = AttentionFlowMatchLayer(hp.hidden_units)

            
            self.match_p_encodes, _ = match_layer.match(self.p_encodes, self.q_encodes,
                                                        self.p_length, self.q_length)

            '''
            #pooling layer
            self.match_p_encodes = \
            tf.keras.layers.MaxPool1D(pool_size=4, strides=None, padding='valid')\
                                    (self.match_p_encodes)


            shape_1 = tf.shape(self.match_p_encodes)
            self.match_p_encodes = tf.reshape(self.match_p_encodes, [-1, hp.p_maxlen, hp.hidden_units])

            shape_2 = tf.shape(self.match_p_encodes)

            self.match_res_shape = (shape_1, shape_2)
            '''
            #normalization
            self.match_p_encodes = tf.layers.batch_normalization(self.match_p_encodes)

            if hp.use_dropout:
                self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

            
            """
            Employs Bi-LSTM again to fuse the context information after match layer
            """
            with tf.variable_scope('fusion'):
                self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                             hp.hidden_units, layer_num=1, concat = False)

                if hp.use_dropout:
                    self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)


            

            """
            Employs Pointer Network to get the the probs of each position
            to be the start or end of the predicted answer.
            Note that we concat the fuse_p_encodes for the passages in the same document.
            And since the encodes of queries in the same document is same, we select the first one.
            """
            
            '''
            with tf.variable_scope('same_question_concat'):
                batch_size = tf.shape(self.start_label)[0]
                concat_passage_encodes = tf.reshape(
                    self.p_encodes,
                    [batch_size, -1, 2 * hp.hidden_units]
                )
                no_dup_question_encodes = tf.reshape(
                    self.q_encodes,
                    [batch_size, -1, 2 * hp.hidden_units]
                )

            '''

            #shape review 2
            #self.reviewer_2 = (tf.shape(concat_passage_encodes), tf.shape(no_dup_question_encodes))

            decoder = PointerNetDecoder(hp.hidden_units)
            self.start_probs, self.end_probs = decoder.decode(self.match_p_encodes,
                                                              self.q_encodes)

            #shape review 3
            #self.reviewer_3 = (tf.shape(self.start_probs), tf.shape(self.start_probs))

            '''
            with tf.name_scope("gama_predict"):
              W = tf.get_variable(
                  "W1",
                  shape=[hp.maxlen * hp.hidden_units, self.gama_span],
                  initializer=tf.contrib.layers.xavier_initializer())

              b = tf.Variable(tf.constant(0.1, shape=[self.gama_span]), name="b1")
              self.l2_loss += tf.nn.l2_loss(W)
              self.l2_loss += tf.nn.l2_loss(b)
              self.gama_prediction = tf.nn.xw_plus_b(self.h_drop, W, b, name="gama predictions")
              #self.preds = tf.argmax(self.scores, 1, name="predictions")
            self.gama_preds = tf.to_int32(tf.argmax(self.gama_prediction, dimension = -1))


            with tf.name_scope('start_predict'):
              W = tf.get_variable(
                  "W2",
                  shape=[hp.maxlen * hp.hidden_units, self.gama_span],
                  initializer=tf.contrib.layers.xavier_initializer())

              b = tf.Variable(tf.constant(0.1, shape=[self.gama_span]), name="b2")
              self.l2_loss += tf.nn.l2_loss(W)
              self.l2_loss += tf.nn.l2_loss(b)
              self.start_prediction = tf.nn.xw_plus_b(self.h_drop, W, b, name="start predictions")

            self.start_pos_preds = tf.to_int32(tf.argmax(self.start_prediction, dimension = -1))



            with tf.name_scope('end_predict'):
              W = tf.get_variable(
                  "W3",
                  shape=[hp.maxlen * hp.hidden_units, self.gama_span],
                  initializer=tf.contrib.layers.xavier_initializer())

              b = tf.Variable(tf.constant(0.1, shape=[self.gama_span]), name="b3")
              self.l2_loss += tf.nn.l2_loss(W)
              self.l2_loss += tf.nn.l2_loss(b)
              self.end_prediction = tf.nn.xw_plus_b(self.h_drop, W, b, name="end predictions")

            self.end_pos_preds = tf.to_int32(tf.argmax(self.end_prediction, dimension = -1))
            '''



                
            if is_training:  
                '''
                self.y_hotting = tf.one_hot(self.y, depth = len(hp.relations))

                #Accuracy
                self.cpl = tf.equal(tf.convert_to_tensor(self.y, tf.int32), self.preds)
                self.cpl = tf.to_int32(self.cpl)
                self.acc = tf.reduce_sum(self.cpl) / tf.to_int32(tf.reduce_sum(self.y_hotting))
                tf.summary.scalar('acc', self.acc)

                # Loss
                #self.y_smoothed = label_smoothing(self.y_hotting)
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_hotting)
                self.mean_loss = (tf.reduce_sum(self.loss) + self.l2_loss*hp.reg_lambda)/tf.reduce_sum(self.y_hotting)
                '''
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
    

