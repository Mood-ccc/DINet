import os
import csv
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score
from DINet.data import MoleDataSet, MoleData, scaffold_split
from DINet.model import DINet
#1.创建目录
def mkdir(path,isdir = True):
    if isdir == False:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok = True)
#2.设置日志记录器
def set_log(name,save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    
    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)
    
    mkdir(save_path)
    
    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)
    
    return log
#3.读取csv文件的表头
def get_header(path):
    with open(path) as file:
        header = next(csv.reader(file))
    
    return header
#4.从文件的表头获得任务名称
def get_task_name(path):
    task_name = get_header(path)[1:]
    
    return task_name
#5.加载csv文件中的数据
def load_data(path,args):
    with open(path) as file:
        reader = csv.reader(file)   #读取csv文件内容
        next(reader)    #跳过表头
        lines = []
        for line in reader:
            lines.append(line)
        data = []
        for line in lines:
            one = MoleData(line,args)    #将每一行数据转换为 MoleData 对象
            data.append(one)
        data = MoleDataSet(data)     #将所有 MoleData 对象组合成一个数据集
        
        fir_data_len = len(data)    #原始数据的长度
        data_val = []
        smi_exist = []
        for i in range(fir_data_len):    #检查数据是由有None
            if data[i].mol is not None:
                smi_exist.append(i)
        data_val = MoleDataSet([data[i] for i in smi_exist])
        now_data_len = len(data_val)    #有效数据的长度
        print('There are ',now_data_len,' smiles in total.')
        if fir_data_len - now_data_len > 0:
            print('There are ',fir_data_len , ' smiles first, but ',fir_data_len - now_data_len, ' smiles is invalid.  ')
        #打印有效数据的数量以及无效数据的数量
    return data_val
#6.将数据集分割为训练集、验证机、测试集
def split_data(data,type,size,seed,log):  #type-分割方式 size-三个列表 seed-随机种子
    assert len(size) == 3
    assert sum(size) == 1
    
    if type == 'random':      #按比例随机分割
        data.random_data(seed)
        train_size = int(size[0] * len(data))
        val_size = int(size[1] * len(data))
        train_val_size = train_size + val_size
        train_data = data[:train_size]
        val_data = data[train_size:train_val_size]
        test_data = data[train_val_size:]
    
        return MoleDataSet(train_data),MoleDataSet(val_data),MoleDataSet(test_data)
    elif type == 'scaffold':     #调用scaffold_split函数进行基于分子结构的分割
        return scaffold_split(data,size,seed,log)
    else:
        raise ValueError('Split_type is Error.')    #如果type两种都不是，报错
#7.获得损失函数
def get_loss():
    #return nn.BCEWithLogitsLoss(reduction='none')    #二元交叉熵损失函数
    return nn.MSELoss()   #MSE
def r_squared(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    mean_label = np.mean(label)
    ss_total = np.sum((label - mean_label) ** 2)
    ss_res = np.sum((label - pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2
def rmse(label,pred):
    result = mean_squared_error(label, pred)
    result = math.sqrt(result)
    return result
def get_metric(metrics):
    metric_functions = {
        'rmse': rmse,
        'r2': r_squared
    }
    return {metric: metric_functions[metric] for metric in metrics}
#10.保存训练好的模型和相关参数
def save_model(path,model,args):
    state = {
        'args':args,
        'state_dict':model.state_dict(),
        'data_scaler':None
        }
    torch.save(state,path)
#11.加载模型
def load_model(path,cuda,log=None,pred_args=None):
    if log is not None:    #如果提供了日志记录器，使用log.debug作为调试函数
        debug = log.debug
    else:
        debug = print   #不然就使用print函数
    
    state = torch.load(path,map_location=lambda storage, loc: storage)    #加载保存的模型状态
    args = state['args']   #提取保存的模型
    
    if pred_args is not None:
        for key,value in vars(pred_args).items():
            if not hasattr(args,key):
                setattr(args, key, value)
    
    state_dict = state['state_dict']
    
    model = DINet(args)
    model_state_dict = model.state_dict()
    
    load_state_dict = {}
    for param in state_dict.keys():
        if param not in model_state_dict:
            debug(f'Parameter is not found: {param}.')
        elif model_state_dict[param].shape != state_dict[param].shape:
            debug(f'Shape of parameter is error: {param}.')
        else:
            load_state_dict[param] = state_dict[param]
            debug(f'Load parameter: {param}.')
    
    model_state_dict.update(load_state_dict)
    model.load_state_dict(model_state_dict)
    
    if cuda:
        model = model.to(torch.device("cuda"))
    
    return model

def load_args(path):
    state = torch.load(path, map_location=lambda storage, loc: storage)
    
    return state['args']

def rmse(label,pred):
    result = mean_squared_error(label,pred)
    result = math.sqrt(result)
    return result


"""

Noam learning rate scheduler with piecewise linear increase and exponential decay.

The learning rate increases linearly from init_lr to max_lr over the course of
the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
Then the learning rate decreases exponentially from max_lr to final_lr over the
course of the remaining total_steps - warmup_steps (where total_steps =
total_epochs * steps_per_epoch). This is roughly based on the learning rate
schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762  ).

"""

class NoamLR(_LRScheduler):
    def __init__(self,optimizer,warmup_epochs,total_epochs,steps_per_epoch,
                 init_lr,max_lr,final_lr):
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        return list(self.lr)

    def step(self,current_step=None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]
