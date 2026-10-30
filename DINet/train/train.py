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

def epoch_train(model,data,loss_f,optimizer,scheduler,args, vocab):   
    model.train()   
    data.random_data(args.seed)   
    loss_sum = 0
    data_used = 0
    iter_step = args.batch_size   
    
    for i in track(range(0,len(data),iter_step),description="Process"):
        if data_used + iter_step > len(data):
            break
        
        data_now = MoleDataSet(data[i:i+iter_step])   
        smile = data_now.smile()
        label = data_now.label()

        mask = torch.Tensor([[x is not None for x in tb] for tb in label])    
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])    
        
        if next(model.parameters()).is_cuda:
            mask, target = mask.cuda(), target.cuda()    
        
        weight = torch.ones(target.shape)   
        if args.cuda:     
            weight = weight.cuda()
        
        model.zero_grad()    
        pred = model(smile)  
        loss = loss_f(pred,target) * weight * mask   
        loss = loss.sum() / mask.sum()      
        loss_sum += loss.item()   
        data_used += len(smile)   
        loss.backward()  
        optimizer.step()   
        if isinstance(scheduler, NoamLR):    
            scheduler.step()
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict(model,data,batch_size):   
    model.eval()    
    pred = []    
    data_total = len(data)   
    
    for i in track(range(0,data_total,batch_size),description="Predict"):
        data_now = MoleDataSet(data[i:i+batch_size])
        smile = data_now.smile()
        
        with torch.no_grad():   
            pred_now = model(smile)  
        
        pred_now = pred_now.data.cpu().numpy()  
                
        pred_now = pred_now.tolist()
        pred.extend(pred_now)    
    
    return pred     

def compute_score(pred,label,metric_f,args,log):   
    info = log.info

    batch_size = args.batch_size
    task_num = args.task_num
    
    if len(pred) == 0:
        return [float('nan')] * task_num   
    
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
        label_val.append(label_val_i)    
    
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


    df = pd.read_csv(args.data_path)
    smiles_list = df['smiles'].tolist()

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
    metric_f = get_metric(args.metrics) 
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

    train_losses = []
    val_losses = []

    early_stopping_patience = args.early_stopping_patience
    early_stopping_delta = args.early_stopping_delta
    early_stopping_counter = 0
    min_epochs = 100 

    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')
        epoch_train(model, train_data, loss_f, optimizer, scheduler, args, vocab=vocab)
        train_pred = predict(model, MoleDataSet(train_data), args.batch_size)
        train_label = train_data.label()
        train_score = compute_score(train_pred, train_label, metric_f, args, log)
        val_pred = predict(model, MoleDataSet(val_data), args.batch_size)
        val_label = val_data.label()
        val_score = compute_score(val_pred, val_label, metric_f, args, log)

        train_loss = np.nanmean(train_score['loss']) if 'loss' in train_score else float('nan')
        val_loss = np.nanmean(val_score['loss']) if 'loss' in val_score else float('nan')

        writer.add_scalar(tag="train-loss", scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag="val-loss", scalar_value=val_loss, global_step=epoch)

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

        main_metric = 'r2'  
        if epoch >= min_epochs: 
            if np.nanmean(val_score[main_metric]) > best_score + early_stopping_delta:
                best_score = np.nanmean(val_score[main_metric])
                best_epoch = epoch
                save_model(os.path.join(args.save_path, 'model.pt'), model, args)
                early_stopping_counter = 0 
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    info(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
                    break

    info(f'Best validation {main_metric} = {best_score:.6f} on epoch {best_epoch}')
    model = DINet(args) 
    if args.cuda:
        model = model.to(torch.device("cuda"))
    model = load_model(os.path.join(args.save_path, 'model.pt'), args.cuda, log)
    test_pred = predict(model, test_data, args.batch_size)
    test_label = test_data.label()
    test_score = compute_score(test_pred, test_label, metric_f, args, log)

    train_pred = predict(model, MoleDataSet(train_data), args.batch_size)
    train_label = train_data.label()
    train_score = compute_score(train_pred, train_label, metric_f, args, log)

    val_pred = predict(model, MoleDataSet(val_data), args.batch_size)
    val_label = val_data.label()
    val_score = compute_score(val_pred, val_label, metric_f, args, log)

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

    return train_score, val_score, test_score, train_losses, val_losses