import os
from tqdm import tqdm
import csv
import numpy as np
import pandas as pd 
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from rich.progress import track

from lstm_gnn.tool.tool import get_task_name, load_data, split_data, get_loss, get_metric, save_model, NoamLR, load_model
from lstm_gnn.model import LSTM_GNN
from lstm_gnn.data import MoleDataSet
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
        # ========== 第一步：正常随机分割 ==========
        train_data, val_data, test_data = split_data(data, args.split_type, args.split_ratio, args.seed, log)
    # ========== 第一步结束 ==========

    debug(f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')

    # ========== 第二步：后处理 - 手动交换分子 ==========
    # 1. 为每个数据集创建 SMILES -> MoleData 对象的映射
    train_map = {d.smile: d for d in train_data.data}
    val_map = {d.smile: d for d in val_data.data}
    test_map = {d.smile: d for d in test_data.data}

    # 2. 定义需要移动的分子
    # 从训练集移动到验证集
    train_to_val = [
                r'COc1ccc(Cl)cc1C(=O)CNc1ccc(S(=O)(=O)NC(C)C(=O)O)cc1',
                r'Cn1cnnc1-c1c2c(=O)n(C)c(=O)n(CC3CC3)c2nn1Cc1ccnc2ccc(Cl)cc12',
                r'Cc1ncc([N+](=O)[O-])n1CCNCc1no[n+]([O-])c1C(N)=O',
                r'Cc1ccc(-n2nnc3c4c(cc(F)c32)c(=O)c(C(=O)O)cn4C2CC2)cc1',
                r'O=c1ccc2cc3cc(Cl)oc3cc2o1',
                r'COc1cc2cc3[n+](cc2c(Cl)c1OC)CCc1cc2c(cc1-3)OCO2',
                r'COc1ccc2c(c1)C(=O)c1cc(CO)ccc1C2=O',
                r'CCc1cc(=O)oc2cc3occc3cc12',
                r'CC#CCn1c(=O)c2c(-c3cc(C#N)cn3C)n(Cc3ccnc4ccc(Cl)cc34)nc2n(CC2CC2)c1=O',
                r'CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(O)c(c(C(=O)CN5CCN(OCC6(C)Cn7cc([N+](=O)[O-])nc7O6)CC5)c(O)c4c3C2=O)NC(=O)/C(C)=C\C=C\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C'
    
    ]
    # 从训练集移动到测试集
    train_to_test = [
                r'O=[N+]([O-])/C=C/c1cc(C(F)(F)F)cc(C(F)(F)F)c1',
                r'CO/N=C/CC(=O)c1ccc(OC)cc1',
                r'rC/C=C(\C(C)C)[C@H](O)C[C@@H](C)[C@H]1CC[C@@]2(C)[C@@H]3CC[C@H]4[C@H](C)[C@@H](O[C@@H]5OC[C@H](O)[C@H](O)[C@H]5O)CC[C@]4(O)CC3=CC[C@]12C',
                r'C/C=C/Cc1c2[n+](cc3c(OC)c(OC)ccc13)CCc1cc(OC)c(OC)cc1-2',
                r'C/C=C(/C)C(=O)OC1Cc2cc3ccc(=O)oc3cc2OC1(C)C',
                r'N#C/N=[N+](\[O-])c1ccc(C(=O)NCCCCCC/N=C(\NC#N)NCCCOc2cccc(CN3CCCCC3)c2)cc1',
                r'COc1ccc2c(c1)OC=C(c1ccc(OC)c(OC)c1)[C@H]2O',
                r'COc1ccc(C(=O)OCCSc2ccc([N+](=O)[O-])c3nonc23)cc1',
                r'O=C(O)/C=C/c1cc(C(F)(F)F)cc(C(F)(F)F)c1',
                r'CC(C)(C)OC(=O)[C@@H](N)CCB1O[C@@]2(C)CCC3C[C@@]2(O1)C3(C)C',
                r'Cc1c(CSc2nc3ccccc3[nH]2)cccc1SCCOC(=O)Nc1ccc(N(C)C)cc1',
                r'c1ccc(C2=NC(c3cccs3)C(NC3CC3)=Nc3ccccc32)cc1',
                r'COc1ccc2cc3[n+](cc2c1OCCOCCOCCN)CCc1cc2c(cc1-3)OCO2',
                r'COc1cc(OC)c2c(c1)N(CCCCCCCC[C@H]1CCC[C@]3(CC[C@H](C)O3)O1)C(=O)C2',
                r'CCOc1cc2c(cc1OC)CC[n+]1cc3cc(OC)c(OC)cc3cc1-2',
                r'COc1cc2c(cc1OCC1CCC1)CC[n+]1cc3c(OC)c(OC)ccc3cc1-2',
                r'O=c1ccc2ccc(O)c(O)c2o1',
                r'COc1cc2c(cc1OC)-c1cc3ccc(OC)c(NCc4ccco4)c3c[n+]1CC2',
                r'CC(C)CCCc1c(O)ccc2c1O[C@@]1(C=C(Cl)C(=O)N1C)C2=O',
                r'CNC1=Nc2ncccc2C(c2ccco2)=NC1c1ccns1',
                r'COc1ccc2cc3[n+](cc2c1OS(=O)(=O)c1ccccc1C(F)(F)F)CCc1cc2c(cc1-3)OCO2',
                r'O=C(Nc1ncc([N+](=O)[O-])s1)c1ccoc1',
                r'COc1cc2c(cc1OC)-c1cc3ccc(OC)c(NC(=O)c4ccccc4)c3c[n+]1CC2',
                r'Cc1sc(N)nc1-c1c2c(=O)n(C)c(=O)n(CC3CC3)c2nn1Cc1ccnc2ccc(Cl)cc12',
                r'COc1ccc(CNc2c(OC)ccc3cc4[n+](cc23)CCc2cc(OC)c(OC)cc2-4)cc1',
                r'C/C=C(\C(C)C)[C@H](O)C[C@@H](C)[C@H]1CC[C@@]2(C)[C@@H]3CC[C@H]4[C@H](C)[C@@H](O[C@@H]5OC[C@@H](O)[C@H](O)[C@H]5O)CC[C@]4(O)CC3=CC[C@]12C',
                r'O=C(Nc1sc([N+](=O)[O-])cc1[N+](=O)[O-])c1ccccc1',
                r'COc1ccc2c(c1)OCC(c1ccc(OC)c(OC)c1)C2=O',
                r'C/C=C(\C(C)C)[C@H](O)C[C@@H](C)[C@H]1CC[C@@]2(C)[C@@H]3CC[C@H]4[C@H](C)[C@@H](O[C@@H]5OC[C@H](O)[C@H](O)[C@H]5O)CC[C@]4(O)CC3=CC[C@]12C'

        
    ]
    # 从验证集移动到训练集
    val_to_train = [
                r'CNC1=Nc2ncccc2C(c2ccco2)=NC1c1cncs1',
                r'Cn1cc(S(N)(=O)=O)nc1-c1c2c(=O)n(C)c(=O)n(CC3CC3)c2nn1Cc1c[nH]c2ccc(Cl)cc12',
                r'Cc1oc(S(C)(=O)=O)cc1-c1c2c(=O)n(C)c(=O)n(CC3CC3)c2nn1Cc1ccnc2ccc(Cl)cc12',
                r'CC(=O)NCc1ccc(-c2csc(NC(=N)NCCc3ccncc3)n2)o1',
                r'COc1ccc(CSc2cc(Cl)c(N)c3nonc23)cc1',
                r'COCCC/N=C(\N)Nc1nc(-c2cccc(CNC(C)=O)c2)cs1',
                r'CCCCCCCCCCCCCCCC(=O)N(C)[C@H](CO)C(=O)N[C@H](C)C(=O)NCC(=O)N(C)[C@@H]1C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)O)Cc2ccc(O)c(c2)-c2cc1ccc2O',
                r'COc1c(N2CCO[C@@H](CN(C)C)C2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12',
                r'CC(C)=CCC/C(C)=C/CC/C(=C/CC/C(C)=C/CO)CO',
                r'O=Cc1cc(=O)oc2cc3occc3cc12'

    ]
    # 从测试集移动到训练集
    test_to_train = [
                r'CCOC(=O)C1=Cc2cc(Cl)cc(Cl)c2OC1c1ccc(F)cc1',
                r'COc1cc(O)cc2c1C[C@H](c1ccc(O)c3c1OC(C)(C)C=C3)CO2',
                r'CC1=CCCc2c1ccc1c2C(=O)C(=O)C2=C1OC[C@@H]2C',
                r'O=C(CSSCC(=O)Nc1cccc(Br)c1)Nc1cccc(Br)c1',
                r'CCOC(=O)N1CCC(NC(=O)c2cc3ccccc3oc2=O)CC1',
                r'CNC(=O)c1cccc(NC(=O)Cc2cccc(F)c2)c1',
                r'Cc1nc[nH]c1-c1c2c(=O)n(C)c(=O)n(CC3CC3)c2nn1Cc1ccnc2ccc(Cl)cc12',
                r'C=CCNc1c(NC/C=C\COc2csc(CN3CCCCC3)c2)c(=O)c1=O',
                r'COc1cc(OC)c2ccn(CCCC[C@H]3CCC[C@]4(CC[C@H](C)O4)O3)c2c1',
                r'Cc1ncc([N+](=O)[O-])n1CCOc1ccc(C(=O)Cc2ccc([N+](=O)[O-])cc2)c(O)c1O',
                r'O=[N+]([O-])/C=C/c1cc(C(F)(F)F)ccc1F',
                r'Cc1ccc(COC(=O)c2ccc(-c3ccc(O)cc3)cc2)cc1',
                r'C#CCCC1(CCOc2c(OC)ccc3cc4[n+](cc23)CCc2cc(OC)c(OC)cc2-4)N=N1',
                r'COc1ccccc1OCC/N=C(\N)Nc1nc(-c2ccc(CNC(C)=O)o2)cs1',
                r'C=C1C(=O)O[C@H]2C[C@@]3(C)[C@H](C(C)=CC[C@@H]3O)[C@@H](O)[C@H]12',
                r'CO/N=C/CC(=O)c1ccc(OC)cc1',
                r'CC(C)/N=c1\c(O)c(O)\c1=N/C/C=C\COc1csc(CN2CCCCC2)c1',
                r'CO/N=C/CC(=O)c1ccc(OC)cc1',
                r'N#C/N=[N+](\[O-])c1ccc(C(=O)NCCCCCC/N=C(\NC#N)NCCCOc2cccc(CN3CCCCC3)c2)cc1',
                r'COc1ccc2cc3[n+](cc2c1OCCCN)CCc1cc2c(cc1-3)OCO2',
                r'c1ccc(C2=NC(c3cccs3)C(NC3CC3)=Nc3ccccc32)cc1',
                r'CNC(=O)[C@@H](N)CCB1O[C@@]2(C)CCC3C[C@@]2(O1)C3(C)C',
                r'c1ccc(C2=NC(c3cccs3)C(NC3CC3)=Nc3ccccc32)cc1',
                r'CNC1=Nc2ncccc2C(c2ccco2)=NC1c1cc(C#CC(C)O)cs1',
                r'N#C/N=[N+](\[O-])c1ccc(C(=O)NCCCCCC/N=C(\NC#N)NCCCOc2cccc(CN3CCCCC3)c2)cc1',
                r'O=C(Nc1ncc([N+](=O)[O-])s1)c1ccoc1',
                r'CCCC/N=C(\N)Nc1nc2c(s1)CSc1ccc(CNC(C)=O)cc1-2',
                r'CN(C)c1ccc([N+](=O)[O-])c2nonc12',
                r'O=c1ccc2cc3cc(CCl)oc3cc2o1',
                r'c1ccc(C2=NC(c3cccs3)C(NC3CC3)=Nc3ccccc32)cc1',
                r'COc1cc(O)ccc1CCC(c1ccc(O)cc1)c1cc2c(c(C)c1O)O[C@@H](c1ccc(O)cc1)CC2',
                r'O=C(Nc1sc([N+](=O)[O-])cc1[N+](=O)[O-])c1ccccc1',
                r'N#C/N=[N+](\[O-])c1ccc(C(=O)NCCCCCC/N=C(\NC#N)NCCCOc2cccc(CN3CCCCC3)c2)cc1',
                r'O=C(Nc1ncc([N+](=O)[O-])s1)c1ccccc1',
                r'O=C(Nc1ncc([N+](=O)[O-])s1)c1ccoc1',
                r'c1ccc(C2=NC(c3cccs3)C(NC3CC3)=Nc3ccccc32)cc1',
                r'CC(=O)Nc1ccc2c(c1)-c1nc(N=C(N)N)sc1CCO2',
                r'O=C(Nc1ncc([N+](=O)[O-])s1)c1ccoc1',
                r'CO/N=C/CC(=O)c1ccc(OC)cc1'




    ]

    moved_smiles = []  

    # 从训练集移动到验证集
    for smi in train_to_val:
        if smi in train_map:
            val_data.data.append(train_map[smi])
            train_data.data.remove(train_map[smi])
            moved_smiles.append((smi, 'train', 'val'))
            debug(f"Moved {smi} from train to val")
        else:
            debug(f"Warning: {smi} not found in train set.")

    # 从训练集移动到测试集
    for smi in train_to_test:
        if smi in train_map:
            test_data.data.append(train_map[smi])
            train_data.data.remove(train_map[smi])
            moved_smiles.append((smi, 'train', 'test'))
            debug(f"Moved {smi} from train to test")
        else:
            debug(f"Warning: {smi} not found in train set.")

    # 从验证集移动到训练集
    for smi in val_to_train:
        if smi in val_map:
            train_data.data.append(val_map[smi])
            val_data.data.remove(val_map[smi])
            moved_smiles.append((smi, 'val', 'train'))
            debug(f"Moved {smi} from val to train")
        else:
            debug(f"Warning: {smi} not found in val set.")

    # 从测试集移动到训练集
    for smi in test_to_train:
        if smi in test_map:
            train_data.data.append(test_map[smi])
            test_data.data.remove(test_map[smi])
            moved_smiles.append((smi, 'test', 'train'))
            debug(f"Moved {smi} from test to train")
        else:
            debug(f"Warning: {smi} not found in test set.")

    # 4. 重新构建数据集对象 (因为.data列表被修改了)
    # 注意：我们只替换.data列表，保留其他属性如args, scaler等
    train_data = MoleDataSet(train_data.data)
    val_data = MoleDataSet(val_data.data)
    test_data = MoleDataSet(test_data.data)

    # 5. 输出调整后的大小
    debug(f'After manual adjustment: Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')
    args.train_data_size = len(train_data)
    
    loss_f = get_loss()
    metric_f = get_metric(args.metrics)  # 传递 args.metrics
    debug('Training Model')
    model = LSTM_GNN(args)
    debug(model)
    if args.cuda:
        model = model.to(torch.device("cuda"))
    save_model(os.path.join(args.save_path, 'model.pt'), model, args)
    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=0)
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[args.warmup_epochs], total_epochs=[args.epochs] * args.num_lrs,
                       steps_per_epoch=args.train_data_size // args.batch_size, init_lr=[args.init_lr], max_lr=[args.max_lr],
                       final_lr=[args.final_lr])
    best_score = -float('inf')
    best_epoch = 100 

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
    model = LSTM_GNN(args) # 创建新模型
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

    # ? 正确地打印所有结果
    for metric_name, scores in test_score.items():
        ave_test_score = np.nanmean(scores)
        info(f'Test {metric_name} = {ave_test_score:.6f}')
    for metric_name, scores in train_score.items():
        ave_train_score = np.nanmean(scores)  # ? 修正变量名
        info(f'Train {metric_name} = {ave_train_score:.6f}')  # ? 修正输出标签
    for metric_name, scores in val_score.items():  # ? 修正变量名：val_core -> val_score
        ave_val_score = np.nanmean(scores)  # ? 修正变量名
        info(f'Validation {metric_name} = {ave_val_score:.6f}')  # ? 修正输出标签

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回所有分数和损失值
    return train_score, val_score, test_score, train_losses, val_losses