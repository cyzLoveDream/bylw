# -*- coding:utf-8 -*-
import os
import sys
from dataset.data_utils import ABSADatasetReader
import torch as t
import torch
import torch.nn as nn
import numpy as np
from torchnet import meter
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from models.deep_mem_net import DeepMemNet
from models.inter_att_net import InterAttNet
from models.lstm import LSTM
from models.td_lstm import TD_LSTM
from models.ram import RAM
from models.cabasc import CABASC
from models.cnn_gate_aspect_model import CNN_Gate_Aspect_Text
from models.myNetWork import MYNETWork
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import warnings
import time
import csv
import gc
from hyperopt import fmin,STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import *
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

class Instrutor:
    def __init__(self,opt):
        self.opt = opt
        print('> training arguments: ')
        for arg in vars(opt):
            print("{0}:{1}".format(arg,getattr(opt,arg)))
        absa_dataset = ABSADatasetReader(dataset=opt.dataset,embed_dim=opt.embed_dim,max_seq_len=opt.max_seq_len,w2v=opt.w2v)
        self.train_data_loader = DataLoader(dataset=absa_dataset.train_data,batch_size=opt.batch_size,shuffle=True)
        self.valid_data_loader = DataLoader(dataset=absa_dataset.valid_data,batch_size=len(absa_dataset.valid_data))
        self.test_data_loader = DataLoader(dataset=absa_dataset.test_data,batch_size=len(absa_dataset.test_data))
        self.model = opt.model_classes(absa_dataset.embedding_matrix, opt).to(device)
        # self.reset_parameters()

    def val(self,model,dataloader):
        """
        计算模型在验证集上的准确率等信息
        """
        model.eval()
        n_test_correct,n_test_total = [],[]
        for t_batch,t_sample_batched in enumerate(dataloader):
            t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
            t_targets = t_sample_batched['polarity'].to(device)
            t_outputs = self.model(t_inputs)
            n_test_correct.extend(np.argmax(np.array(t_outputs.data.tolist()),1))
            n_test_total.extend(t_targets.data.tolist())
        test_acc = accuracy_score(np.array(n_test_total),np.array(n_test_correct))
        test_f1 = f1_score(np.array(n_test_total), np.array(n_test_correct),average="macro")
        test_precise = precision_score(np.array(n_test_total),np.array(n_test_correct),average="macro")
        test_recall = recall_score(np.array(n_test_total),np.array(n_test_correct),average="macro")
        model.train()
        return test_acc,test_f1, test_precise, test_recall

    def run(self):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        params = filter(lambda x: x.requires_grad,self.model.parameters())
        # optimizer
        optimizer = self.opt.optimizer(params,lr=self.opt.learning_rate,weight_decay=self.opt.weight_decay)
        global_step = 0
        # meter
        loss_meter = meter.AverageValueMeter()
        curMax = float("-inf")
        count = 0
        model_name = ""
        train_cur = 0
        test_cur= 0
        val_cur = 0
        test_f1_cur, test_precise_cur, test_recall_cur, test_acc_cur = 0,0,0,0
        print("begin train....")
        epoch_count = 0
        for epoch in range(self.opt.num_epoch):
            epoch_count += 1
            loss_meter.reset()
            for i_batch,sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(device) for col in self.opt.inputs_cols]
                targets = sample_batched["polarity"].to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs,targets)
                loss.backward()
                optimizer.step()
                # print(i_batch ,loss)
            # early stop in 10
            val_acc,_,_,_ = self.val(self.model,self.valid_data_loader)
            train_acc,_,_,_ = self.val(self.model,self.train_data_loader)
            if epoch_count % 5 == 0:
                test_acc,test_f1, test_precise, test_recall = self.val(self.model,self.test_data_loader)
                print("the train acc is {0}, the val acc is {1}, the test acc is {2}".format(train_acc,val_acc,test_acc))
            test_acc,test_f1, test_precise, test_recall = self.val(self.model, self.test_data_loader)
            if val_acc > curMax:
                curMax = val_acc
                train_cur, val_cur = train_acc, val_acc
                test_f1_cur, test_precise_cur, test_recall_cur, test_acc_cur = test_f1, test_precise, test_recall,test_acc
                count = 0
                # model_name = self.model.save()
            else:
                count += 1
            if count >= 10:
                break
        del self.model
        del self.opt
        gc.collect()
        print("the train acc is {0}, the val acc is {1}, the test acc is {2}".format(train_cur,curMax,test_acc_cur))
        return test_f1_cur, test_precise_cur, test_recall_cur, test_acc_cur

def obejective(params):
    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'ian': InterAttNet,
        'memnet': DeepMemNet,
        'ram': RAM,
        'cabasc': CABASC,
        'cnn': CNN_Gate_Aspect_Text,
        'my': MYNETWork
        }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices','text_right_with_aspect_indices'],
        'ian': ['text_raw_indices','aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices','aspect_indices','text_left_with_aspect_indices'],
        'ram': ['text_raw_indices','aspect_indices'],
        'cabasc': ['text_raw_indices','aspect_indices','text_left_with_aspect_indices',
                   'text_right_with_aspect_indices'],
        'cnn': ['text_raw_indices','aspect_indices'],
        'my': ['text_left_indices','text_right_indices','aspect_indices']
        }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
        }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        }
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    for k,v in params.items():
        if k in ['batch_size','hidden_dim','max_seq_len','fillter']:
            v = int(v)
        setattr(opt,k,v)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    opt.model_classes = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = device
    ins = Instrutor(opt)
    train_acc,test_acc,test_f1 = ins.run()
    loss = 1 - test_acc
    # Write to the csv file ('a' means append)
    of_connection = open(opt.out_file,'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss,test_acc,test_f1,params])
    of_connection.close()
    return {"loss": loss, "test-acc": test_acc,"test-f1":test_f1, "params":params,"status":STATUS_OK}

def tiaocan():
    for d in ["my-restaurant"]:
        print("-------------------------------------%s---------------------------------------------------" %d)
        for m in ["ram"]:
            out_file = './trials' + "_{0}_{1}".format(d,m) + ".csv"
            of_connection = open(out_file,'w')
            writer = csv.writer(of_connection)
            # Write the headers to the file
            writer.writerow(['loss','test-acc','test-f1','params'])
            of_connection.close()
            params = dict(model_name="td_lstm", dataset="my-laptop",
                          initializer=hp.choice("initializer", ["xavier_normal_", "xavier_uniform_", "orthogonal_"]),
                          learning_rate=hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)), num_epoch=100,
                          batch_size=hp.quniform("batch_size", 16, 128, 2), embed_dim=300,
                          hidden_dim=hp.quniform("hidden_dim", 100, 500, 10),
                          max_seq_len=hp.quniform("max_seq_len", 10, 80, 1), polarity_dim=3, hops=3, plot_every=50,
                          seed=1024, weight_decay=hp.loguniform("weight_decay", np.log(0.001), np.log(0.1)),
                          fillter=hp.quniform("fillter", 100, 500, 10),
                          kernel_sizes=hp.choice("kernel_sizes", [[1, 2, 3], [1, 2], [2, 3], [1, 3]]), aspect_nums=0,
                          optimizer=hp.choice("optimizer",
                                              ["adadelta", 'adagrad', 'adam', 'adamax', 'asgd', 'rmsprop', 'sgd']))
            params["model_name"] = m
            params["dataset"] = d
            params["out_file"] = out_file
            bayes_trials = Trials()
            best = fmin(fn=obejective, space=params, algo=tpe.suggest, max_evals=50,trials=bayes_trials)
            bayes_trials_results = sorted(bayes_trials.results,key=lambda x: x['loss'])
            result = []
            result_f1 = []
            for res in bayes_trials_results[:5]:
                result.append(res.get("test-acc"))
                result_f1.append(res.get("test-f1"))
            del bayes_trials_results
            del bayes_trials
            del best
            gc.collect()
            print("the model is {0} the mean test-acc is {1} the mean test-f1 is {2}".format(m,np.mean(np.array(result), axis=0), np.mean(np.array(result_f1))))

def result():
    for m in ["ian","ram","cnn","my"]:
        print("--------------------%s-------------------"%m)
        params = {
            "model_name": m,
            "dataset": "bylw",
            "initializer": "xavier_uniform_",
            "learning_rate": 0.005,
            "num_epoch": 100,
            "batch_size": 64,
            "embed_dim": 300,
            "hidden_dim": 50,
            "max_seq_len": 55,
            "polarity_dim": 3,
            "hops": 3,
            "plot_every": 50,
            "seed": 42,
            "weight_decay": 0.001,
            "fillter": 390,
            "kernel_sizes": (2,3),
            "aspect_nums": 0,
            "optimizer": 'adam',
            "w2v":"cbow",
            }
        model_classes = {
            'lstm': LSTM,
            'td_lstm': TD_LSTM,
            'ian': InterAttNet,
            'memnet': DeepMemNet,
            'ram': RAM,
            'cabasc': CABASC,
            'cnn': CNN_Gate_Aspect_Text,
            'my': MYNETWork
            }
        input_colses = {
            'lstm': ['text_raw_indices'],
            'td_lstm': ['text_left_with_aspect_indices','text_right_with_aspect_indices'],
            'ian': ['text_raw_indices','aspect_indices'],
            'memnet': ['text_raw_without_aspect_indices','aspect_indices','text_left_with_aspect_indices'],
            'ram': ['text_raw_indices','aspect_indices'],
            'cabasc': ['text_raw_indices','aspect_indices','text_left_with_aspect_indices',
                       'text_right_with_aspect_indices'],
            'cnn': ['text_raw_indices','aspect_indices'],
            'my': ['text_left_indices','text_right_indices','aspect_indices']
            }
        initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal,
            'orthogonal_': torch.nn.init.orthogonal_,
            }
        optimizers = {
            'adadelta': torch.optim.Adadelta,  # default lr=1.0
            'adagrad': torch.optim.Adagrad,  # default lr=0.01
            'adam': torch.optim.Adam,  # default lr=0.001
            'adamax': torch.optim.Adamax,  # default lr=0.002
            'asgd': torch.optim.ASGD,  # default lr=0.01
            'rmsprop': torch.optim.RMSprop,  # default lr=0.01
            'sgd': torch.optim.SGD,
            }
        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
        for k,v in params.items():
            if k in ['batch_size','hidden_dim','max_seq_len','fillter']:
                v = int(v)
            setattr(opt,k,v)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        opt.model_classes = model_classes[opt.model_name]
        opt.inputs_cols = input_colses[opt.model_name]
        opt.initializer = initializers[opt.initializer]
        opt.optimizer = optimizers[opt.optimizer]
        opt.device = device
        ins = Instrutor(opt)
        test_f1_cur, test_precise_cur, test_recall_cur, test_acc_cur = ins.run()
        print("the test recall is {0} the test-precise is {1} the test-f1 is {2}".format(test_recall_cur, test_precise_cur, test_f1_cur))
if __name__ == '__main__':
    # Hyper Parameters
    # tiaocan()
    warnings.filterwarnings("ignore")
    result()

    # model_classes = {
    #     'lstm': LSTM,
    #     'td_lstm': TD_LSTM,
    #     'ian': InterAttNet,
    #     'memnet': DeepMemNet,
    #     'ram': RAM,
    #     'cabasc': CABASC,
    #     'cnn': CNN_Gate_Aspect_Text,
    #     'my': MYNETWork
    #     }
    # input_colses = {
    #     'lstm': ['text_raw_indices'],
    #     'td_lstm': ['text_left_with_aspect_indices','text_right_with_aspect_indices'],
    #     'ian': ['text_raw_indices','aspect_indices'],
    #     'memnet': ['text_raw_without_aspect_indices','aspect_indices','text_left_with_aspect_indices'],
    #     'ram': ['text_raw_indices','aspect_indices'],
    #     'cabasc': ['text_raw_indices','aspect_indices','text_left_with_aspect_indices',
    #                'text_right_with_aspect_indices'],
    #     'cnn': ['text_raw_indices','aspect_indices'],
    #     'my': ['text_left_indices','text_right_indices','aspect_indices']
    #     }
    # initializers = {
    #     'xavier_uniform_': torch.nn.init.xavier_uniform_,
    #     'xavier_normal_': torch.nn.init.xavier_normal,
    #     'orthogonal_': torch.nn.init.orthogonal_,
    #     }
    # optimizers = {
    #     'adadelta': torch.optim.Adadelta,  # default lr=1.0
    #     'adagrad': torch.optim.Adagrad,  # default lr=0.01
    #     'adam': torch.optim.Adam,  # default lr=0.001
    #     'adamax': torch.optim.Adamax,  # default lr=0.002
    #     'asgd': torch.optim.ASGD,  # default lr=0.01
    #     'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    #     'sgd': torch.optim.SGD,
    #     }
    # best_result = {}
    # maxACC = float("-inf")
    # best_params = None
    # for i in optimizers.keys():
    #     opt = parser.parse_args()
    #     params["optimizer"] = i
    #     for k,v in params.items():
    #         setattr(opt,k,v)
    #     torch.manual_seed(opt.seed)
    #     torch.cuda.manual_seed(opt.seed)
    #     np.random.seed(opt.seed)
    #     opt.model_classes = model_classes[opt.model_name]
    #     opt.inputs_cols = input_colses[opt.model_name]
    #     opt.initializer = initializers[opt.initializer]
    #     opt.optimizer = optimizers[opt.optimizer]
    #     opt.device = device
    #     ins = Instrutor(opt)
    #     if opt.dataset not in ["my-laptop","my-restaurant"]:
    #         train_acc,val_acc,test_acc = ins.run()
    #         best_result[i] = [train_acc, val_acc, test_acc]
    #         if test_acc > maxACC:
    #             maxACC = test_acc
    #             best_params = i
    #         print("the params is：{0} the result is {1},{2},{3}".format(i,train_acc,val_acc,test_acc))
    #     else:
    #         train_acc, test_acc,test_f1 = ins.run()
    #         best_result[i] = [train_acc,test_acc,test_f1]
    #         if test_acc > maxACC:
    #             maxACC = test_acc
    #             best_params = i
    #         print("the params is：{0} the test result acc is {1} and the test f1 is {2} ".format(i,test_acc, test_f1))
    # print("the best param is {0}, the result is {1},{2},{3}".format(best_params,best_result[best_params][0],
    #                                                                 best_result[best_params][1],
    #                                                                 best_result[best_params][2]))



