import pandas as pd
import numpy as np
import os
import warnings
import time
warnings.filterwarnings("ignore")
import jieba
jieba.load_userdict("../raw_data/user_dict.txt")
from gensim.models.word2vec import Word2Vec
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
model = Word2Vec.load("../cache/w2v_cbow")

class Tokenizer(object):
    """
    将语料序列化
    """
    def __init__(self,max_seq_len=None,max_aspect_len=None,char_level=None):
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.char_level = char_level
        self.word2ix = {}
        self.ix2word = {}
        self.ix = 1

    def fit_on_text(self,text):
        """
        所有文本形成的一个超长字符串
        :param text:
        :return:
        """
        for t in text:
            words = jieba.lcut(t)
            for w in words:
                if w not in self.word2ix:
                    self.word2ix[w] = self.ix
                    self.ix2word[self.ix] = w
                    self.ix += 1

    @staticmethod
    def pad_sequence(sequence,maxlen,dtype="int64",padding="pre",truncating="pre",value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == "pre":
            # 前向截断
            trunc = sequence[-maxlen:]
        else:
            # 后向截断
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc,dtype=dtype)
        if padding == "post":
            # 后向填充
            x[:len(trunc)] = trunc
        else:
            # 前向填充
            x[-len(trunc):] = trunc
        return x.tolist()

    def text_to_sequence(self,text,reverse=False):
        words = jieba.lcut(text)
        unknowix = len(self.word2ix) + 1
        sequence = [self.word2ix[w] if w in self.word2ix else unknowix for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = "post"
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence,self.max_seq_len,dtype="int64",padding=pad_and_trunc,
                                      truncating=pad_and_trunc)

def get_data(fname):
    fin = open(fname, 'r', encoding="utf-8", newline="\n", errors="ignore")
    lines = fin.readlines()
    all_data = []
    for i in range(0, len(lines), 3):
        text_left,_, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i+1].strip()
        polarity = lines[i+2].strip()
        text = text_left + " " + aspect + " " + text_right
        words = jieba.lcut(text)
        sequence = np.zeros(300)
        for w in words:
            if w in model:
                sequence += model[w]
        text_raw_seq = sequence / len(words)
        polarity = int(polarity)
        data = {
            "text_raw_seq": text_raw_seq,
            "polarity":polarity
        }
        all_data.append(data)
    pls = []
    for i in all_data:
        pls.append(i.get("polarity"))
    print("the all data is {0}, the POS is {1}, the NEU is {2}, the NEG is {3}".format(len(all_data), pls.count(2),pls.count(1),pls.count(0)))
    return all_data

def prepare_data(fname):
    all_data = get_data(fname=fname)
    data = []
    label = []
    for d in all_data:
        data.append(d.get("text_raw_seq"))
        label.append(d.get("polarity"))
    return np.array(data), np.array(label)

def get_allData():
    tokenizer = Tokenizer()
    data = pd.read_csv("../data_set/w2v_raw_crops.csv")
    data = data["body"].tolist()
    tokenizer.fit_on_text(data)
    print("begin handle data....")
    now = time.time()
    train, train_label = prepare_data("../data_set/train.csv")
    print("get the train data: ", time.time() - now)
    now = time.time()
    valid, valid_label = prepare_data("../data_set/valid.csv")
    print("get the valid data: ", time.time() - now)
    now = time.time()
    test, test_label = prepare_data("../data_set/test.csv")
    print("get the test data: ", time.time() - now)
    return train, train_label, valid, valid_label, test, test_label

if __name__ == '__main__':
    get_allData()



