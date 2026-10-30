import os
from tqdm import tqdm
import csv
import numpy as np
import pandas as pd 
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from rich.progress import track

from DINet.tool.tool import get_task_name, load_data, split_data, get_loss, get_metric, save_model, NoamLR, load_model
from DINet.model import DINet
from DINet.data import MoleDataSet
from sklearn.metrics import mean_squared_error, r2_score

def build_vocab(smiles_list):
    all_chars = []
    for smi in smiles_list:
        all_chars.extend(list(smi))
    counter = Counter(all_chars)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (char, _) in enumerate(counter.most_common()):
        vocab[char] = i + 2
    return vocab
    
from collections import Counter

def epoch_train(model,data,loss_f,optimizer,scheduler,args, vocab):    #执行一个训练周期
    model.train()     #设置为训练模式
    data.random_data(args.seed)    #使用 args.seed 对数据进行随机化处理
    loss_sum = 0
    data_used = 0
    iter_step = args.batch_size    #迭代步长为args.batch_size
    
    for i in track(range(0,len(data),iter_step),description="Process"):
        if data_used + iter_step > len(data):
            break
        
        data_now = MoleDataSet(data[i:i+iter_step])    #提取数据
        smile = data_now.smile()
        label = data_now.label()

        mask = torch.Tensor([[x is not None for x in tb] for tb in label])     #创建掩码矩阵
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])     #将标签中的None换为0
        
        if next(model.parameters()).is_cuda:
            mask, target = mask.cuda(), target.cuda()     #如果在GPU上运行，将掩码和目标张量移动到GPU
        
        weight = torch.ones(target.shape)    #创建权重矩阵，初始值为1
        if args.cuda:      #如果使用GPU,将权重矩阵移动到GPU
            weight = weight.cuda()
        
        model.zero_grad()     #清空模型梯度
        pred = model(smile)   #使用模型预测
        loss = loss_f(pred,target) * weight * mask    #计算损失值，考虑权重和掩码
        loss = loss.sum() / mask.sum()        #计算平均损失值
        loss_sum += loss.item()    #累加损失值
        data_used += len(smile)    #更新已处理数据量
        loss.backward()    #反向传播计算梯度
        optimizer.step()    #更新模型参数
        if isinstance(scheduler, NoamLR):      #更新学习率
            scheduler.step()
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict(model,data,batch_size):    #数据预测
    model.eval()     #将模式设置为评估模式
    pred = []     #初始化预测结果列表
    data_total = len(data)    #数据集总大小
    
    for i in track(range(0,data_total,batch_size),description="Predict"):
        data_now = MoleDataSet(data[i:i+batch_size])
        smile = data_now.smile()
        
        with torch.no_grad():    #禁用梯度计算，减少内存占用
            pred_now = model(smile)   #使用模型进行预测
        
        pred_now = pred_now.data.cpu().numpy()   #将预测结果转换为 NumPy 数组，再转换为列表
                
        pred_now = pred_now.tolist()
        pred.extend(pred_now)    #将当前批次的预测结果添加到总预测结果中
    
    return pred     #返回预测结果

def compute_score(pred,label,metric_f,args,log):    #计算预测结果的评估指标
    info = log.info

    batch_size = args.batch_size
    task_num = args.task_num
    
    if len(pred) == 0:
        return [float('nan')] * task_num   #如果预测结果为空，返回包含NaN的列表
    
    pred_val = []
    label_val = []
    for i in range(task_num):
        pred_val_i = []
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)    #遍历每个任务，提取有效的预测值和标签值
    
    results = {}
    for metric_name, metric_func in metric_f.items():
        results[metric_name] = []
        for i in range(task_num):
            if len(label_val[i]) == 0:
                info(f'Warning: No valid labels for task {i}.')
                results[metric_name].append(float('nan'))
                continue
            re = metric_func(label_val[i], pred_val[i])
            results[metric_name].append(re)
            
    results['loss'] = []
    for i in range(task_num):
        if len(label_val[i]) > 0:
            mse = mean_squared_error(label_val[i], pred_val[i])
            results['loss'].append(mse)
        else:
            results['loss'].append(float('nan'))

    return results
def fold_train(args, log, writer):
    info = log.info
    debug = log.debug

    debug('Start loading data')

    # 加载 SMILES 数据
    df = pd.read_csv(args.data_path)
    smiles_list = df['smiles'].tolist()
    # 构建词汇表
    vocab = build_vocab(smiles_list)
    args.vocab_size = len(vocab)
    args.task_names = get_task_name(args.data_path)
    data = load_data(args.data_path, args)
    args.task_num = data.task_num()
    if args.task_num > 1:
        args.is_multitask = 1

    debug(f'Splitting dataset with Seed = {args.seed}.')
    if args.val_path:
        val_data = load_data(args.val_path, args)
    if args.test_path:
        test_data = load_data(args.test_path, args)
    if args.val_path and args.test_path:
        train_data = data
    elif args.val_path:
        split_ratio = (args.split_ratio[0], 0, args.split_ratio[2])
        train_data, _, test_data = split_data(data, args.split_type, split_ratio, args.seed, log)
    elif args.test_path:
        split_ratio = (args.split_ratio[0], args.split_ratio[1], 0)
        train_data, val_data, _ = split_data(data, args.split_type, split_ratio, args.seed, log)
    else:
        train_data, val_data, test_data = split_data(data, args.split_type, args.split_ratio, args.seed, log)

    debug(f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')

    args.train_data_size = len(train_data)

    loss_f = get_loss()
    metric_f = get_metric(args.metrics)  # 传递 args.metrics
    debug('Training Model')
    model = DINet(args)
    debug(model)
    if args.cuda:
        model = model.to(torch.device("cuda"))
    save_model(os.path.join(args.save_path, 'model.pt'), model, args)
    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=0)
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[args.warmup_epochs], total_epochs=[args.epochs] * args.num_lrs,
                       steps_per_epoch=args.train_data_size // args.batch_size, init_lr=[args.init_lr], max_lr=[args.max_lr],
                       final_lr=[args.final_lr])
    best_score = -float('inf')
    best_epoch = 0

    # 初始化损失值列表
    train_losses = []
    val_losses = []

    # 初始化早停机制的变量
    early_stopping_patience = args.early_stopping_patience
    early_stopping_delta = args.early_stopping_delta
    early_stopping_counter = 0
    min_epochs = 100  # 至少训练的 epoch 数量

    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')
        epoch_train(model, train_data, loss_f, optimizer, scheduler, args, vocab=vocab)
        train_pred = predict(model, MoleDataSet(train_data), args.batch_size)
        train_label = train_data.label()
        train_score = compute_score(train_pred, train_label, metric_f, args, log)
        val_pred = predict(model, MoleDataSet(val_data), args.batch_size)
        val_label = val_data.label()
        val_score = compute_score(val_pred, val_label, metric_f, args, log)

        # 计算训练和验证的平均损失
        train_loss = np.nanmean(train_score['loss']) if 'loss' in train_score else float('nan')
        val_loss = np.nanmean(val_score['loss']) if 'loss' in val_score else float('nan')

        # 将损失值写入 TensorBoard
        writer.add_scalar(tag="train-loss", scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag="val-loss", scalar_value=val_loss, global_step=epoch)

        # 记录损失值
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        info(f'Train Loss: {train_loss:.6f}')
        info(f'Validation Loss: {val_loss:.6f}')

        for metric_name, scores in train_score.items():
            ave_train_score = np.nanmean(scores)
            info(f'Train {metric_name} = {ave_train_score:.6f}')
            writer.add_scalar(tag=f"train-{metric_name}", scalar_value=ave_train_score, global_step=epoch)
        for metric_name, scores in val_score.items():
            ave_val_score = np.nanmean(scores)
            info(f'Validation {metric_name} = {ave_val_score:.6f}')
            writer.add_scalar(tag=f"val-{metric_name}", scalar_value=ave_val_score, global_step=epoch)
        if args.task_num > 1:
            for metric_name, scores in val_score.items():
                for one_name, one_score in zip(args.task_names, scores):
                    info(f'Validation {one_name} {metric_name} = {one_score:.6f}')

        main_metric = 'r2'  # 主要的性能指标，如 R2
        if epoch >= min_epochs:  # 只有在训练了至少 min_epochs 个 epoch 之后，才开始检查早停条件
            if np.nanmean(val_score[main_metric]) > best_score + early_stopping_delta:
                best_score = np.nanmean(val_score[main_metric])
                best_epoch = epoch
                save_model(os.path.join(args.save_path, 'model.pt'), model, args)
                early_stopping_counter = 0  # 重置早停计数器
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    info(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
                    break

    info(f'Best validation {main_metric} = {best_score:.6f} on epoch {best_epoch}')
    model = DINet(args) # 使用相同的 args 和 vocab 创建新模型
    if args.cuda:
        model = model.to(torch.device("cuda"))
    model = load_model(os.path.join(args.save_path, 'model.pt'), args.cuda, log) # 只加载权重
    test_pred = predict(model, test_data, args.batch_size)
    test_label = test_data.label()
    test_score = compute_score(test_pred, test_label, metric_f, args, log)

    train_pred = predict(model, MoleDataSet(train_data), args.batch_size)
    train_label = train_data.label()
    train_score = compute_score(train_pred, train_label, metric_f, args, log)

    val_pred = predict(model, MoleDataSet(val_data), args.batch_size)
    val_label = val_data.label()
    val_score = compute_score(val_pred, val_label, metric_f, args, log)

    # 正确地打印所有结果
    for metric_name, scores in test_score.items():
        ave_test_score = np.nanmean(scores)
        info(f'Test {metric_name} = {ave_test_score:.6f}')
    for metric_name, scores in train_score.items():
        ave_train_score = np.nanmean(scores)
        info(f'Train {metric_name} = {ave_train_score:.6f}')
    for metric_name, scores in val_score.items():
        ave_val_score = np.nanmean(scores)
        info(f'Validation {metric_name} = {ave_val_score:.6f}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回所有分数和损失值
    return train_score, val_score, test_score, train_losses, val_losses