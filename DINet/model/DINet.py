import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
from DINet.data import GetPubChemFPs
from torch_geometric.data import InMemoryDataset, DataLoader
import torch.nn.functional as F

from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_add_pool, global_max_pool

import networkx as nx
from torch_geometric import data as DATA

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

atts_out = []

#提取分子指纹并使用LSTM处理
class FP_LSTM(nn.Module):
    def __init__(self, args):
        super(FP_LSTM, self).__init__()
        self.fp_dim = 4049  # 指纹总维度
        self.hidden_dim = args.hidden_size
        self.fp_2_dim = args.fp_2_dim
        self.dropout_fpn = args.dropout

        # 定义网络层
        self.LSTM = nn.LSTM(1, self.hidden_dim, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, 1)  # 将 LSTM 输出的维度从 hidden_dim*2 转换为 1
        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)  # 将指纹维度从 4049 转换为 fp_2_dim
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)  # 将 fp_2_dim 转换为 hidden_dim
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smile):
        fp_list = []
        for i, one in enumerate(smile):
            fp = []
            mol = Chem.MolFromSmiles(one)
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            fp_pubcfp = GetPubChemFPs(mol)
            fp_Avalon = pyAvalonTools.GetAvalonFP(mol)
            fp_ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fp.extend(fp_maccs)
            fp.extend(fp_phaErGfp)
            fp.extend(fp_pubcfp)
            fp.extend(fp_Avalon)
            fp.extend(fp_ecfp)
            fp_list.append(fp)

        fp_list = torch.Tensor(fp_list)
        if self.cuda:
            fp_list = fp_list.cuda()
        fp_list = fp_list.unsqueeze(-1)  # (batch_size, 4049, 1)

        fpn_lstm, (hn, cn) = self.LSTM(fp_list)
        fpn_list = self.fc(fpn_lstm)
        fp_list = fpn_list.squeeze()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)  # 输出维度为 hidden_dim
        return fpn_out

#将SMILES字符串转换为图神经网络的输入格式，为GNN提供输入
class GNN_Dataset(InMemoryDataset):
    #转换为图的三个组成部分
    def __init__(self, smile):
        super(GNN_Dataset, self).__init__()
        data_list = []   #初始化表格，存储转换后的图数据对象
        for item in smile:    #循环处理每个字符串
            c_size, features, edge_index = self.smile_to_graph(item)
            # make the graph ready for PyTorch Geometrics GCN algorithms:创建图数据对象
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(
                                    edge_index).transpose(1, 0),
                                )
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)  #将图数据对象添加到data_list
        self.data, self.slices = self.collate(data_list)   #将多个图数据对象组合成一个批量数据
#提取单个原子的特征向量
    def atom_features(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetHybridization(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetChiralTag(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])
#转换为独热编码向量
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                "input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
#将字符串转换为图的组成部分
    def smile_to_graph(self, smile):
        mol = Chem.MolFromSmiles(smile)

        c_size = mol.GetNumAtoms()
        #提取节点特征
        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature / sum(feature))  #归一化
        #提取边索引
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges.append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])
        #构建图结构
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        features = np.array(features)      #转换为特征矩阵
        return c_size, features, edge_index    #节点数量、节点特征矩阵、边索引列表
        
class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')  # 使用 add 聚合
        # 消息函数：[x_j, x_j - x_i]
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i):
        # 构造消息：[x_j, x_j - x_i]
        edge_msg = x_j - x_i
        msg_input = torch.cat([x_j, edge_msg], dim=1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out):
        return aggr_out


class MPNNModel(nn.Module):
    def __init__(self, num_features=100, hidden_dim=64, output_dim=300, num_layers=2, dropout=0.2):
        super(MPNNModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 简单嵌入
        self.embed = nn.Linear(num_features, hidden_dim)

        # 多层 MPNN
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 全连接头
        self.fc_g1 = nn.Linear(hidden_dim * 2, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embed(x)

        # 消息传递
        for layer in self.mpnn_layers:
            x = layer(x, edge_index)
            x = self.relu(x)

        # 图池化：gap + gmp 拼接
        x_add = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_add, x_max], dim=1)

        # 使用和 GAT-GCN 一样的回归头结构
        x_graph = self.relu(self.fc_g1(x_graph))
        x_graph = self.dropout_layer(x_graph)
        x_graph = self.fc_g2(x_graph)

        return x_graph

#将字符串转换为图数据，并通过指定GNN模型前向传播，返回最终输出
class GNN_Model(nn.Module):
    def __init__(self, dropout, hidden_size):
        super(GNN_Model, self).__init__()
        self.cuda = 'cuda:0'
        self.model = MPNNModel(num_features=100, hidden_dim=hidden_size, output_dim=hidden_size, dropout=dropout)

    def forward(self, smile):
        data = GNN_Dataset(smile)
        loader = DataLoader(data, batch_size=len(smile), shuffle=True)
        for batch_idx, data in enumerate(loader):
            data = data.to(self.cuda)
            output = self.model(data)
        return output

class FeatureFusionAttention(nn.Module):
    def __init__(self, in_dim):  # in_dim = 2 * hidden_size
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),  # 降维（防过拟合）
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim),   # 恢复到 in_dim
            nn.Sigmoid()                      # 输出 0~1 权重
        )

    def forward(self, x):  # x: [B, in_dim]
        weights = self.attn_net(x)  # [B, in_dim]
        return x * weights          # [B, in_dim]


class DINet_Model(nn.Module):
    def __init__(self, args, cuda, dropout_fpn):
        super(DINet_Model, self).__init__()
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        self.myenc = GNN_Model(dropout_fpn, hidden_size=args.hidden_size)  # GNN编码器
        self.encoder2 = FP_LSTM(args)  # LSTM编码器
        self.feature_attn = FeatureFusionAttention(in_dim=2 * args.hidden_size)
        self.residual = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size)  # 残差连接
        self.sigmoid = nn.Sigmoid()
        self.create_scale(args)  # 初始化特征缩放层
        self.create_ffn(args)  # 初始化前馈神经网络
        self.reduce_dim = nn.Linear(2 * args.hidden_size, args.hidden_size)
        
    def create_gat(self, args):
        # 初始化GNN模型
        self.myenc = GNN_Model(dropout=args.dropout, hidden_size=args.hidden_size)

    def create_fpn(self, args):
        # 初始化LSTM模型
        self.encoder2 = FP_LSTM(args)

    def create_scale(self, args):
        self.fc_gat = nn.Linear(args.hidden_size, args.hidden_size)  # GNN特征缩放
        self.fc_fpn = nn.Linear(args.hidden_size, args.hidden_size)  # LSTM特征缩放
        self.act_func = nn.ReLU()  # 激活函数

    def create_ffn(self, args):
        linear_dim = args.hidden_size
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim * 2, out_features=linear_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
        )

    def forward(self, input):
        gat_out = self.myenc(input)  # GNN编码器的输出
        fpn_out = self.encoder2(input)  # LSTM编码器的输出

        # 特征缩放
        gat_out = self.fc_gat(gat_out)
        gat_out = self.act_func(gat_out)
        fpn_out = self.fc_fpn(fpn_out)
        fpn_out = self.act_func(fpn_out)

        # 注意力加权融合
        combined = torch.cat([gat_out, fpn_out], dim=1)  # 将两种特征拼接
        attn_output = self.feature_attn(combined) 

        # 残差连接
        residual = self.residual(combined)  # 残差连接
        output = attn_output + residual  # 注意力加权结果与残差相加

        # 传入前馈神经网络生成最终结果
        output = self.ffn(output)
        return output

# 初始化模型
def DINet(args, vocab=None):
    model = DINet_Model(args, args.cuda, args.dropout)
    model.create_gat(args)
    model.create_fpn(args)
    model.create_scale(args)
    model.create_ffn(args)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model