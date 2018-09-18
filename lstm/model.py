#-*-coding:utf8-*-
import tensorflow as tf
import pandas as pd

class lstm_char(object):
    def __init__(self, class_size, vocab, vocab_size, embedding_size, rnn_size, num_layers, hidden_layer_size):
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.class_size = class_size

        #input placeholder
        self.inputs_passage = tf.placeholder(tf.int32, [None, None], name='inputs_passage')  #json passage
        self.inputs_query = tf.placeholder(tf.int32, [None, None], name='inputs_query')  #json query
        self.label_y = tf.placeholder(tf.int32, [None,class_size], name="targets") #finall answer
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.inputs_passage_length = tf.placeholder(tf.int32, [None,],name='input_passage_length')
        self.inputs_query_length = tf.placeholder(tf.int32, [None,], name='input_query_length')

        logits = self.encoder_concat()
        self.predicted = tf.argmax(logits,1)
        self.loss = tf.losses.softmax_cross_entropy(self.label_y,logits)
        correct_predicted = tf.equal(self.predicted, tf.argmax(self.label_y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_predicted, tf.float32))
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = opt.minimize(self.loss,global_step=tf.train.get_global_step)


    def encoder_layer(self,input_data, num_layers, input_query_length,name):
        '''
        encoder layer
        :param input_data:
        :param num_layers:
        :param input_query_length:
        :return:
        '''
        with tf.name_scope(name):
            encoder_embed_input =tf.contrib.layers.embed_sequence(input_data, self.vocab_size, self.embedding_size)
            cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(i) for i in self.rnn_size])
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=input_query_length, dtype=tf.float32,scope=name)
            return encoder_output, encoder_state

    def lstm_cell(self):
        '''
        :return: lstm cell
        '''
        lstm_cell_ = tf.contrib.rnn.LSTMCell(self.rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1))
        return lstm_cell_

    def encoder_concat(self):
        encoder_output_passage, encoder_state_passage = self.encoder_layer(self.inputs_passage,self.num_layers, self.inputs_passage_length,'passage')
        encoder_output_query, encoder_state_query = self.encoder_layer(self.inputs_query, self.num_layers, self.inputs_query_length,'query')
        encoder_state_concat = tf.concat([encoder_state_passage, encoder_state_query], 1)
        hidden_layer = tf.layers.dense(encoder_state_concat, self.hidden_layer_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer(0.1))
        logits = tf.layers.dense(hidden_layer, self.class_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.constant_initializer(0.1))
        return logits