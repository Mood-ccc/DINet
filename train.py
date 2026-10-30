import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from DINet.train import fold_train
from DINet.tool import set_log, set_train_argument, mkdir
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from DINet.model import DINet
from DINet.tool import load_data, split_data, set_train_argument, set_log
from DINet.data import MoleDataSet
import optuna

import warnings
warnings.filterwarnings('ignore')

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

comment = input("运行名称")
writer = SummaryWriter(comment=comment)

# 设置随机种子
def training(args, log):
    info = log.info
    
    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path
    
    # 初始化一个字典来存储每个评估指标的结果
    scores = {metric: [] for metric in args.metrics}
    
    # 只训练一次，使用指定的随机种子
    info(f'Seed {args.seed}')
    args.save_path = os.path.join(save_path, f'Seed_{args.seed}')
    mkdir(args.save_path)
        
    # fold_train 函数返回一个字典，包含每个评估指标的结果
    train_score, val_score, test_score, train_losses, val_losses = fold_train(args, log, writer)
        
    # 将每个评估指标的结果存储到对应的列表中
    for metric in args.metrics:
            scores[metric].append(test_score[metric])
    
    # 将每个评估指标的结果转换为 NumPy 数组
    for metric in args.metrics:
        scores[metric] = np.array(scores[metric])
    
    info(f'Running 1 fold in total.')
    
    # 计算每个评估指标的平均值和标准差
    for metric in args.metrics:
        ave_task_score = np.nanmean(scores[metric], axis=1)
        score_ave = np.nanmean(ave_task_score)
        score_std = np.nanstd(ave_task_score)
        info(f'Average test {metric} = {score_ave:.6f} +/- {score_std:.6f}')
        
        if args.task_num > 1:
            for i, one_name in enumerate(args.task_names):
                info(f'    average all-fold {one_name} {metric} = {np.nanmean(scores[metric][:, i]):.6f} +/- {np.nanstd(scores[metric][:, i]):.6f}')
    
    return {metric: (np.nanmean(scores[metric]), np.nanstd(scores[metric])) for metric in args.metrics}

if __name__ == '__main__':
    args = set_train_argument()
    log = set_log('train',args.log_path)
    training(args,log) 