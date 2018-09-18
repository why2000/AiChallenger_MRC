#-*-coding:utf8-*-
import tensorflow as tf

class QAnet(object):
    def __init__(self,config):
        self.config = config

        # input
        self.input_context = tf.placeholder(tf.int32, [None,config['context_length']],'context')
        self.input_question = tf.placeholder(tf.int32, [None, config['question_length']],'question')
        self.input_answer_1 = tf.placeholder(tf.int32, [None, config['answer_set_length']],'answer_sel1')
        self.input_answer_2 = tf.placeholder(tf.int32, [None, config['answer_set_length']], 'answer_sel2')
        self.input_answer_3 = tf.placeholder(tf.int32, [None, config['answer_set_length']], 'answer_sel3')
        self.input_y = tf.placeholder(tf.int32,[None, config['answer_length']],'answer')
        self.input_y_index = tf.placeholder(tf.int64, [None], 'answer_index')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.embedding()
        encode_context = self.encoder(self.embedded_context_expanded,self.config['context_length'])
        encode_question = self.encoder(self.embedded_question_expanded, self.config['question_length'], reuse=True)
        self.encode_answer_1 = self.encoder(self.embedded_answer_1_expanded, self.config['answer_set_length'],reuse=True)
        self.encode_answer_2 = self.encoder(self.embedded_answer_2_expanded, self.config['answer_set_length'], reuse=True)
        self.encode_answer_3 = self.encoder(self.embedded_answer_3_expanded, self.config['answer_set_length'], reuse=True)
        info_encode = tf.concat([encode_context,encode_question,self.encode_answer_1,self.encode_answer_2,self.encode_answer_3],1)

        self.answer_encode = self.encoder(self.embedded_y_expanded,self.config['answer_length'])
        self.h_drop = tf.nn.dropout(info_encode,self.dropout_keep_prob)
        self.output()
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.answer_encode, logits=self.output_vec)
        self.train()
        tf.summary.scalar('loss', self.loss)
        self.accuary = self.acc()
        self.summary()


    def embedding(self):
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.config['vocab_size'], self.config['embedding_size']], -1.0, 1.0), name="W_embedding")
            self.embedded_context = tf.nn.embedding_lookup(self.W, self.input_context)
            self.embedded_question = tf.nn.embedding_lookup(self.W, self.input_question)
            self.embedded_answer_1 = tf.nn.embedding_lookup(self.W, self.input_answer_1)
            self.embedded_answer_2 = tf.nn.embedding_lookup(self.W, self.input_answer_2)
            self.embedded_answer_3 = tf.nn.embedding_lookup(self.W, self.input_answer_3)
            self.embedded_y = tf.nn.embedding_lookup(self.W, self.input_y)

            self.embedded_context_expanded = tf.expand_dims(self.embedded_context, -1)
            self.embedded_question_expanded = tf.expand_dims(self.embedded_question, -1)
            self.embedded_answer_1_expanded = tf.expand_dims(self.embedded_answer_1, -1)
            self.embedded_answer_2_expanded = tf.expand_dims(self.embedded_answer_2, -1)
            self.embedded_answer_3_expanded = tf.expand_dims(self.embedded_answer_3, -1)
            self.embedded_y_expanded = tf.expand_dims(self.embedded_y, -1)

    def encoder(self, encoder_tensor,sequence_length,reuse = None):
        pooled_outputs = []
        for i, filter_size in enumerate(self.config['filter_sizes']):
            scope = "conv-maxpool-" + str(filter_size)
            with tf.variable_scope(scope, reuse=reuse):
                # Convolution Layer
                filter_shape = [filter_size, self.config['embedding_size'], 1, self.config['num_filters']]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                b = tf.Variable(tf.constant(0.1, shape=[self.config['num_filters']]), name="b")
                conv = tf.nn.conv2d(encoder_tensor, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        self.num_filters_total = self.config['num_filters'] * len(self.config['filter_sizes'])
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        return h_pool_flat

    def output(self):
        with tf.name_scope('output'):
            W = tf.get_variable(
                "W",
                shape=[self.num_filters_total * 5, self.num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters_total]), name="b")
            self.output_vec = tf.nn.xw_plus_b(self.h_drop, W, b, name='output_vec')

    def distance(self, input1, input2):
        dis = tf.sqrt(tf.reduce_sum(tf.square(input1-input2),1))
        return tf.reshape(dis,[-1,1])

    def acc(self):
        dis = []
        dis.append(self.distance(input1=self.output_vec,input2=self.encode_answer_1))
        dis.append(self.distance(input1=self.output_vec,input2=self.encode_answer_2))
        dis.append(self.distance(input1=self.output_vec, input2=self.encode_answer_3))
        dis_all = tf.concat(dis,1)
        predict = tf.arg_min(dis_all,1)
        correct_prediction = tf.equal(self.input_y_index,predict)
        acc = tf.reduce_mean(tf.cast(correct_prediction,'float'), name="acc")
        tf.summary.scalar('acc', acc)
        return acc


    def train(self):
        train_opt = tf.train.AdamOptimizer(self.config['learning_rate'])
        with tf.device('/gpu:0'):
            self.train_step = train_opt.minimize(self.loss)

    def summary(self):
        merged = tf.summary.merge_all()

