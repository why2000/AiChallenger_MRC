#-*-coding:utf8-*-
import numpy as np
import json

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


class Datainput(object):
    def __init__(self, vocab ,json_list_all,query_length,passage_length,answer_length, answer_sel_size):
        self.vocab = vocab
        self.json_list_all = json_list_all
        self.query_length = query_length
        self.passage_length = passage_length
        self.answer_length = answer_length
        self.answer_sel_size = answer_sel_size

    def get_rand_query(self, json_data):
        query_char = [self.vocab.index(char)+1 for char in json_data['query']]
        return query_char

    def get_rand_passage(self,json_data):
        passage = [self.vocab.index(char)+1 for char in json_data['passage']]
        return passage

    def get_rand_y(self,json_data):
        y = [self.vocab.index(char)+1 for char in json_data['answer']]
        return y

    def padd_rand_y(self,x):
        if len(x) <= self.answer_length:
            return x + [0]*(self.answer_length-len(x))
        return x[:self.answer_length]

    def padd_rand_query(self,x):
        if len(x) <= self.query_length:
            return x + [0]*(self.query_length-len(x))
        return x[:self.query_length]

    def padd_rand_passage(self,x):
        if len(x) <= self.passage_length:
            return x + [0]*(self.passage_length-len(x))
        return x[:self.passage_length]

    def get_rand_y_index(self, json_data):
        answer_set = json_data['alternatives'].split('|')
        return answer_set.index(json_data['answer'])

    def answer_set(self, json_datas):
        answer_sel1 = []
        answer_sel2 = []
        answer_sel3 = []
        for json_data in json_datas:
            answer_set = json_data['alternatives'].split('|')
            answer_sel1_ = [self.vocab.index(char)+1 for char in answer_set[0]]
            answer_sel1.append(self.padd_rand_y(answer_sel1_))

            answer_sel2_ = [self.vocab.index(char) + 1 for char in answer_set[1]]
            answer_sel2.append(self.padd_rand_y(answer_sel2_))

            answer_sel3_ = [self.vocab.index(char) + 1 for char in answer_set[2]]
            answer_sel3.append(self.padd_rand_y(answer_sel3_))
        return answer_sel1, answer_sel2, answer_sel3

    def get_batch_data(self,batch_size):
        rand_index = np.random.choice(len(self.json_list_all), size=batch_size)
        rand_ = np.array(self.json_list_all)[rand_index]
        rand_query = map(self.get_rand_query, rand_)
        rand_query_padd = map(self.padd_rand_query, rand_query)
        rand_passage = map(self.get_rand_passage, rand_)
        rand_passage_padd = map(self.padd_rand_passage, rand_passage)
        rand_y = map(self.get_rand_y, rand_)
        rand_y_padd = map(self.padd_rand_y, rand_y)
        rand_y_index = map(self.get_rand_y_index, rand_)
        answer_sel1, answer_sel2, answer_sel3 = self.answer_set(rand_)
        return rand_query_padd, rand_passage_padd, rand_y_padd, rand_y_index,answer_sel1,answer_sel2,answer_sel3
    
    def get_batch(self,batch_size):
        rand_index = np.random.choice(len(self.json_list_all), size=batch_size)
        rand_ = np.array(self.json_list_all)[rand_index]
        rand_query_padd = []
        rand_passage_padd = []
        rand_y_padd = []
        rand_y_index = []
        answer_sel1 = []
        answer_sel2 = []
        answer_sel3 = []
        answer_sel1, answer_sel2, answer_sel3 = self.answer_set(rand_)
        for rand_json in rand_:
            rand_query = self.get_rand_query(rand_json)
            rand_query_padd.append(self.padd_rand_query(rand_query))
            rand_passage = self.get_rand_passage(rand_json)
            rand_passage_padd.append(self.padd_rand_passage(rand_passage))
            rand_y = self.get_rand_y(rand_json)
            rand_y_padd.append(self.padd_rand_y(rand_y))
            rand_y_index.append(self.get_rand_y_index(rand_json))
        return rand_query_padd, rand_passage_padd, rand_y_padd, rand_y_index,answer_sel1,answer_sel2,answer_sel3


