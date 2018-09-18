#-*-coding:utf8-*-
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from untils import json_data_until
from model import lstm_char
import tensorflow as tf


train_data_path = '/disk/private-data/shh/AIChallenger/data/ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
test_data_path = '/disk/private-data/shh/AIChallenger/data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
vali_data_path = '/disk/private-data/shh/AIChallenger/data/ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'

train_data = json_data_until.json_data_loader(train_data_path)
vali_data = json_data_until.json_data_loader(vali_data_path)
vocab = json_data_until.get_vocab_char([train_data, vali_data])

train_step = 3000
batch_size = 100
learning_rate = 0.001
embedding_size = 512
rnn_size = [64, 128]
num_layers = 2
hidden_layer_size = 128

train_data_set = json_data_until.batch_data(vocab,train_data)
vali_data = json_data_until.batch_data(vocab, vali_data)

model_lstm_ = lstm_char(3,vocab,len(vocab),embedding_size,rnn_size,num_layers, hidden_layer_size)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(train_step):
    input_query, input_passage, input_y, input_query_len, input_passage_len = train_data_set.get_batch(batch_size)
    if i%100 == 0:
        loss, acc = sess.run([model_lstm_.loss,model_lstm_.acc], feed_dict={model_lstm_.inputs_query:input_query, model_lstm_.inputs_passage:input_passage
                                               , model_lstm_.inputs_query_length:input_query_len, model_lstm_.inputs_passage_length:input_query_len
                                                   ,model_lstm_.label_y:input_y, model_lstm_.learning_rate:learning_rate})
        print("step:{}, train acc:{}, loss:{}".format(i,acc,loss))
    train_op = sess.run(model_lstm_.train_op, feed_dict={model_lstm_.inputs_query:input_query, model_lstm_.inputs_passage:input_passage
                                               , model_lstm_.inputs_query_length:input_query_len, model_lstm_.inputs_passage_length:input_query_len
                                                   ,model_lstm_.label_y:input_y, model_lstm_.learning_rate:learning_rate})