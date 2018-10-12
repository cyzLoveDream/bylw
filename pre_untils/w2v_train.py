import pandas as pd
import numpy as np
import json
import jieba
import warnings
import gensim
import time
import logging
import os.path
import sys
##训练word2vec模型
#获取日志信息
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
warnings.filterwarnings("ignore")
jieba.load_userdict("../raw_data/user_dict.txt")
def read_data(file_path):
    data = pd.read_csv(file_path)
    texts = data["body"].tolist()
    texts_list = list(map(lambda x: jieba.lcut(x), texts))
    return texts_list


def train(sentences, dim=300,windows_count=5,mn_count=3,sg=0,model_file="../cache/w2v",model_w2v_file=""):
    """
    
    :param sentences: 
    :param dim: 
    :param windows_count: 
    :param mn_count: 
    :param sg: 0: CBOW, 1: SKip-gram
    :return: 
    """
    model = gensim.models.Word2Vec(sentences=sentences, size=dim, window=windows_count,
                                   min_count=mn_count,seed=1024,workers=4, iter=100,sg=sg)
    # print("begin train...")
    # now = time.time()
    # model.train(sentences)
    # print("finish train: ", time.time() - now)
    model.wv.save_word2vec_format(model_w2v_file,binary=True)
    model.save(model_file)

def get_w2vmat(model_file, w2v_format_file):
    model = gensim.models.Word2Vec.load(model_file)
    s_list =[]
    s = ""
    with open(w2v_format_file, "w+", encoding="utf-8") as wf:
        wf.write(str(len(s_list))+ str(300))
        for i in model.wv.vocab:
            s += i
            s += " "
            s += str(model[i])
            wf.write(s + "\n")


if __name__ == '__main__':
    file_path = "../data_set/w2v_raw_crops.csv"
    texts_list = read_data(file_path)
    print("get raw crops: ", len(texts_list))
    train(texts_list, model_file="../cache/w2v_cbow",model_w2v_file="../cache/w2v_cbow_format")
    train(texts_list,sg=1, model_file="../cache/w2v_skip",model_w2v_file="../cache/w2v_skip_format")
    # get_w2vmat("../cache/w2v_cbow","../cache/w2v_cbow_fromat")
