from argparse import ArgumentParser
import torch
from .tool import mkdir
import optuna
#设置训练参数
def add_train_argument(p):
    p.add_argument('--data_path',type=str,
                   help='The path of input CSV file.')
    p.add_argument('--save_path',type=str,default='model_save',
                   help='The path to save output model.pt.,default is "model_save/"')
    p.add_argument('--log_path',type=str,default='log',
                   help='The dir of output log file.')
    p.add_argument('--is_multitask',type=int,default=0,
                   help='Whether the dataset is multi-task. 0:no  1:yes.')
    p.add_argument('--task_num',type=int,default=1,
                   help='The number of task in multi-task training.')
    p.add_argument('--split_type',type=str,choices=['random', 'scaffold'],default='random',
                   help='The type of data splitting.')
    p.add_argument('--split_ratio',type=float,nargs=3,default=[0.8,0.1,0.1],
                   help='The ratio of data splitting.[train,valid,test]')
    p.add_argument('--val_path',type=str,
                   help='The path of excess validation data.')
    p.add_argument('--test_path',type=str,
                   help='The path of excess testing data.')
    p.add_argument('--seed',type=int,default=2,
                   help='The random seed of model. Using in splitting data.')
    p.add_argument('--num_folds',type=int,default=5,
                   help='The number of folds in cross validation.')
    #p.add_argument('--metric',type=str,choices=['auc', 'prc-auc'],default=None,
                   #help='The metric of data evaluation.')
    p.add_argument('--metric', type=str, nargs='+', default=['rmse', 'r2'],
                   help='The metric(s) of data evaluation. Can be one or more of ["auc", "prc-auc", "rmse", "r2"].')
    p.add_argument('--epochs',type=int,default=5,
                   help='The number of epochs.')
    p.add_argument('--batch_size',type=int,default=32,
                   help='The size of batch.')
    p.add_argument('--hidden_size',type=int,default=300,
                   help='The dim of hidden layers in model.')
    p.add_argument('--lstm_layers',type=int,default=5,
                   help='The dim of hidden layers in model.')
    p.add_argument('--fp_2_dim',type=int,default=512,
                   help='The dim of the second layer in fpn.')
    p.add_argument('--dropout',type=float,default=0.5,
                   help='The dropout of gcn.')
    p.add_argument('--early_stopping_patience', type=int, default=100,
                   help='Number of epochs with no improvement after which training will be stopped.')
    p.add_argument('--early_stopping_delta', type=float, default=0.001,
                   help='Minimum change in the monitored quantity to qualify as an improvement.')

def set_train_argument():
    p = ArgumentParser()
    add_train_argument(p)
    args = p.parse_args()
    
    assert args.data_path
    
    mkdir(args.save_path)
    
    if args.metric is None:
        #args.metric = 'auc'
        args.metric = ['rmse', 'r2']
    
    args.metrics = args.metric

    args.cuda = torch.cuda.is_available()
    args.init_lr = 0.002088875506398324 # 1e-4    #初始学习率
    args.max_lr = 1e-3            #最大学习率
    args.final_lr = 1e-4          #最终学习率
    args.warmup_epochs = 2.0      #预热轮数
    args.num_lrs = 1              #学习率数量
    
    return args
"""
def objective(trial):
    params = {
     #   'hidden_size': trial.suggest_int('hidden_size', 100, 500),
      #  'fp_2_dim': trial.suggest_int('fp_2_dim', 256, 1024),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'init_lr': trial.suggest_loguniform('init_lr', 1e-5, 1e-3),
        'max_lr': trial.suggest_loguniform('max_lr', 1e-4, 1e-2),
        'final_lr': trial.suggest_loguniform('final_lr', 1e-5, 1e-3),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
        'batch_size': trial.suggest_int('batch_size', 16, 60),
    }
    return params
"""
