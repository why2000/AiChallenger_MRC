#-*-coding:utf8-*-
import pandas as pd
import json
import pickle
import numpy as np


def json_data_loader(json_data_path):
    '''
    json数据文件解析
    :param json_data_path:  exp:'/disk/private-data/......'
    :return: list of json   exp: [{},{}.....]
    '''
    loader_f = open(json_data_path, 'r', encoding='utf8')
    lines = loader_f.readlines()
    json_list = []
    for line in lines:
        json_list.append(json.loads(line.rstrip('\n')))
    return json_list

def get_json_char(json_):
    query = [i for i in json_['query']]
    passage = [i for i in json_['passage']]
    alternatives = [i for i in json_['alternatives']]
    char_total = query + passage + alternatives
    char_set = list(set(char_total))
    return char_set

def get_vocab_char(json_list_total):
    vocab = []
    for json_list in json_list_total:
        json_char_total = map(get_json_char, json_list)
        for json_char in json_char_total:
            vocab += json_char
    vocab = list(set(vocab))
    return vocab

class batch_data(object):
    def __init__(self,vocab, json_data_total):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.json_data_total = json_data_total

    def get_rand_query(self, json_data):
        query_char = [self.vocab.index(char) for char in json_data['query']]
        return query_char

    def get_rand_passage(self,json_data):
        passage = [self.vocab.index(char) for char in json_data['passage']]
        return passage

    def get_rand_y(self,json_data):
        answer = json_data['answer']
        alternatives = json_data.split('|')
        if answer == alternatives[0]:
            return [1, 0, 0]
        elif answer == alternatives[1]:
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    def padd_rand_query(self,x):
        return x + [self.vocab_size]*(self.max_query_len - len(x))

    def padd_rand_passage(self,x):
        return x + [self.vocab_size]*(self.max_passage_len - len(x))
    
    def get_query_len(x):
        return len(x)

    def get_batch(self,bach_size):
        rand_index = np.random.choice(len(self.json_data_total), size=bach_size)
        rand_ = self.json_data_total[rand_index]
        rand_query = map(self.get_rand_query, rand_)
        self.max_query_len = np.argmax(rand_query,1)
        rand_query_x =  map(self.padd_rand_query,rand_query)
        rand_query_len = map(self.get_query_len, rand_query)
        
        rand_passage = map(self.get_rand_passage, rand_)
        self.max_passage_len = np.argmax(rand_passage,1)
        rand_passage_x = map(self.padd_rand_passage, rand_passage)
        rand_passage_len = map(self.get_query_len, rand_passage)
        
        rand_y = map(self.get_rand_y, rand_)
        return rand_query, rand_passage, rand_y, rand_query_len, rand_passage_len


if __name__ == "__main__":
    train_data_path_ = '../data/ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
    test_data_path_ = '../data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    vali_data_path_ = '../data/ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'
    json_list_all = []
    json_list_all.append(json_data_loader(train_data_path_))
    json_list_all.append(json_data_loader(test_data_path_))
    json_list_all.append(json_data_loader(vali_data_path_))
    vocab = get_vocab_char(json_list_all)
    pickle.dump(vocab,open('word_vocab.pkl','wb'))