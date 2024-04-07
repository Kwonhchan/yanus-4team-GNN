import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv

class NGCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, layers, heads):
        super(NGCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.emb_size = emb_size
        self.GC_layers = nn.ModuleList()
        self.GAT_layers = nn.ModuleList()
        self.GraphSAGE_layers = nn.ModuleList()
        self.Cheb_layers = nn.ModuleList()
        self.MLP_layers = nn.Sequential(
            nn.Linear(emb_size * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # 여기서 최종 출력 차원은 512입니다.
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.batch_norm_layers = nn.ModuleList()

        # MLP_layers를 통과한 후의 차원이 512임을 반영
        input_dim = 512  # MLP 후 출력 차원으로 시작
        
        # GCN 층 추가
        for output_dim in layers:
            self.GC_layers.append(GCNConv(input_dim, output_dim))
            self.batch_norm_layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim  # 다음 레이어를 위해 입력 차원 업데이트

        # GraphSAGE 층 추가
        self.GraphSAGE_layers.append(SAGEConv(input_dim, input_dim))
        self.Cheb_layers.append(ChebConv(input_dim, input_dim, K=2))

        # GAT 층 추가
        for _ in range(heads):
            # GAT의 concat=False 설정에 따라 출력 차원은 input_dim과 동일합니다.
            self.GAT_layers.append(GATConv(input_dim, input_dim, heads, concat=False))

        self.prediction_layer = nn.Linear(input_dim, 1)

    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch
        
        user_indices = data.x[:, 0].long()
        item_indices = data.x[:, 1].long()

        u_emb = self.user_embedding(user_indices)
        i_emb = self.item_embedding(item_indices)

        # 임베딩 병합
        x = torch.cat([u_emb, i_emb], dim=1)
        x = self.MLP_layers(x)
        
        # GCN & GraphSAGE 층을 통과
        for gc_layer, bn_layer in zip(self.GC_layers, self.batch_norm_layers):
            x = F.relu(bn_layer(gc_layer(x, edge_index)))
        x = F.relu(self.GraphSAGE_layers[0](x, edge_index))
        x = F.relu(self.Cheb_layers[0](x, edge_index))

        # GAT 층을 통과
        for gat_layer in self.GAT_layers:
            x = F.elu(gat_layer(x, edge_index))

        # 최종 예측 점수 계산
        scores = self.prediction_layer(x).squeeze()

        return scores
























# #NGCF 발전 버전1
# class NGCF(nn.Module):
#     def __init__(self, num_users, num_items, emb_size, layers, heads):
#         super(NGCF, self).__init__()
#         self.user_embedding = nn.Embedding(num_users, emb_size)
#         self.item_embedding = nn.Embedding(num_items, emb_size)
#         self.emb_size = emb_size
#         self.GC_layers = nn.ModuleList()
#         self.GAT_layers = nn.ModuleList()

#         input_dim = emb_size * 2  # 사용자와 아이템 임베딩을 연결(concatenate)
        
#         # GCN 층 추가
#         for output_dim in layers:
#             self.GC_layers.append(GCNConv(input_dim, output_dim))
#             input_dim = output_dim  # 다음 레이어를 위해 입력 차원 업데이트

#         # GAT 층 추가
#         for _ in range(heads):
#             self.GAT_layers.append(GATConv(input_dim, input_dim, heads=1, concat=True))
#             # GAT 출력은 헤드 수에 따라 차원이 증가하지만 여기서는 concat을 False로 하여 차원 유지

#         self.prediction_layer = nn.Linear(input_dim, 1)

#     def forward(self, data):
#         edge_index, batch = data.edge_index, data.batch
        
#         user_indices = data.x[:, 0].long()
#         item_indices = data.x[:, 1].long()

#         u_emb = self.user_embedding(user_indices)
#         i_emb = self.item_embedding(item_indices)

#         # 임베딩 병합
#         x = torch.cat([u_emb, i_emb], dim=1)
        
#         # GCN 층을 통과
#         for gc_layer in self.GC_layers:
#             x = F.relu(gc_layer(x, edge_index))
        
#         # GAT 층을 통과
#         for gat_layer in self.GAT_layers:
#             x = F.elu(gat_layer(x, edge_index))

#         # 최종 예측 점수 계산
#         scores = self.prediction_layer(x).squeeze()

#         return scores

































# # 아이템 추천 NGCF 모델 정의
# import torch
# import torch.nn.functional as F
# from torch.nn import Linear
# from torch_geometric.nn import GCNConv

# class NGCF(nn.Module):
#     def __init__(self, num_users, num_items, emb_size, layers):
#         super(NGCF, self).__init__()
#         self.user_embedding = torch.nn.Embedding(num_users, emb_size)
#         self.item_embedding = torch.nn.Embedding(num_items, emb_size)
#         self.emb_size = emb_size
#         self.GC_layers = torch.nn.ModuleList()
        
#         input_dim = emb_size * 2  # 사용자와 아이템 임베딩을 연결(concatenate)
#         for output_dim in layers:
#             self.GC_layers.append(GCNConv(input_dim, output_dim))
#             input_dim = output_dim  # 다음 레이어를 위해 입력 차원 업데이트

#         # 마지막 GCNConv 레이어의 출력을 사용하여 예측을 수행하는 선형 레이어
#         self.prediction_layer = Linear(input_dim, 1)  

#     def forward(self, data):
#         # data는 PyTorch Geometric의 Batch 객체
#         edge_index, batch = data.edge_index, data.batch
        
#         # 사용자와 아이템 인덱스 추출
#         user_indices = data.x[:, 0].long()
#         item_indices = data.x[:, 1].long()

#         # 사용자와 아이템 임베딩 계산
#         u_emb = self.user_embedding(user_indices)
#         i_emb = self.item_embedding(item_indices)

#         # 임베딩 병합
#         x = torch.cat([u_emb, i_emb], dim=1)
        
#         # GCN 레이어를 통과시키며 임베딩 업데이트
#         for gc_layer in self.GC_layers:
#             x = F.relu(gc_layer(x, edge_index))

#         # 각 사용자-아이템 쌍에 대한 예측 점수 계산
#         scores = self.prediction_layer(x).squeeze()

#         return scores
