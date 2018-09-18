import numpy as np
import pandas as pd
import model
import data_until
import argparse
import tensorflow as tf


train_data_path = '/disk/private-data/shh/AIChallenger/data/ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
test_data_path = '/disk/private-data/shh/AIChallenger/data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
vali_data_path = '/disk/private-data/shh/AIChallenger/data/ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'

json_list_all = []
json_list_all.append(data_until.json_data_loader(train_data_path))
json_list_all.append(data_until.json_data_loader(test_data_path))
json_list_all.append(data_until.json_data_loader(vali_data_path))
vocab = data_until.get_vocab_char(json_list_all)

train_step = 20000
batch_size = 200

query_length = 30
passage_length = 500
answer_length = 5
answer_sel_size = 3

Data_input_train = data_until.Datainput(vocab,json_list_all[0],query_length, passage_length, answer_length, answer_sel_size)
Data_input_vali = data_until.Datainput(vocab, json_list_all[2],query_length, passage_length, answer_length, answer_sel_size)

config = {
    "context_length": passage_length,
    "question_length": query_length,
    "answer_set_length": answer_length,
    "answer_length":answer_length,
    "vocab_size": len(vocab),
    "embedding_size": 1024,
    "filter_sizes": [1,2,3],
    "num_filters": 500,
    "learning_rate": 0.001,
}

QaModel = model.QAnet(config)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(train_step):
    rand_query_padd, rand_passage_padd, rand_y_padd, rand_y_index, answer_sel1, answer_sel2, answer_sel3 = Data_input_train.get_batch(batch_size)
    if i%10 == 0:
        train_loss,train_acc = sess.run([QaModel.loss,QaModel.accuary],feed_dict={
            QaModel.input_context:rand_passage_padd,
            QaModel.input_question:rand_query_padd,
            QaModel.input_answer_1:answer_sel1,
            QaModel.input_answer_2: answer_sel2,
            QaModel.input_answer_3: answer_sel3,
            QaModel.input_y: rand_y_padd,
            QaModel.input_y_index: rand_y_index,
            QaModel.dropout_keep_prob:1.0
        })
        print("step {}, train acc:{}, train loss:{}".format(i,train_acc,train_loss))
        
        rand_query_padd_test, rand_passage_padd_test, rand_y_padd_test, rand_y_index_test, answer_sel1_test, answer_sel2_test, answer_sel3_test = Data_input_vali.get_batch(1000)
        test_loss,test_acc = sess.run([QaModel.loss,QaModel.accuary],feed_dict={
            QaModel.input_context:rand_passage_padd_test,
            QaModel.input_question:rand_query_padd_test,
            QaModel.input_answer_1:answer_sel1_test,
            QaModel.input_answer_2: answer_sel2_test,
            QaModel.input_answer_3: answer_sel3_test,
            QaModel.input_y: rand_y_padd_test,
            QaModel.input_y_index: rand_y_index_test,
            QaModel.dropout_keep_prob:1.0
        })
        print("step {}, test acc:{}, test loss:{}".format(i,test_acc,test_loss))
        

    train_step = sess.run(QaModel.train_step,feed_dict={
        QaModel.input_context: rand_passage_padd,
        QaModel.input_question: rand_query_padd,
        QaModel.input_answer_1: answer_sel1,
        QaModel.input_answer_2: answer_sel2,
        QaModel.input_answer_3: answer_sel3,
        QaModel.input_y: rand_y_padd,
        QaModel.input_y_index: rand_y_index,
        QaModel.dropout_keep_prob: 0.5
    })
    



