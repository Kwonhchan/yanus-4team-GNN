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




# 아이템 추천 NGCF 모델 정의
class NGCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, layers):
        super(NGCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.emb_size = emb_size
        self.GC_layers = nn.ModuleList()
        self.Bi_layers = nn.ModuleList()
        for From, To in zip(layers[:-1], layers[1:]):
            self.GC_layers.append(nn.Linear(From, To))
            self.Bi_layers.append(nn.Linear(From, To))

    def forward(self, user_indices, item_indices, adj_matrices):
        predictions = []
        for b in range(len(adj_matrices)):
            adj_matrix = adj_matrices[b].to_dense()
            
            u_emb = self.user_embedding(user_indices[b])  # [num_users_in_batch, emb_size]
            i_emb = self.item_embedding(item_indices[b])  # [num_items_in_batch, emb_size]

            # 여기서 dim=0으로 사용자와 아이템 임베딩을 올바르게 결합합니다.
            ego_embeddings = torch.cat([u_emb, i_emb], dim=0)  # 수정됨: dim=1 -> dim=0

            all_embeddings = [ego_embeddings]

            for gc_layer, bi_layer in zip(self.GC_layers, self.Bi_layers):
                side_embeddings = torch.matmul(adj_matrix, ego_embeddings)
                sum_embeddings = F.leaky_relu(gc_layer(side_embeddings))
                bi_embeddings = F.leaky_relu(bi_layer(ego_embeddings * side_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                all_embeddings.append(ego_embeddings)

            all_embeddings = torch.cat(all_embeddings, dim=1)
            # 올바른 차원으로 분할하기 위해 dim=0을 사용
            u_g_embeddings, i_g_embeddings = torch.split(ego_embeddings, [u_emb.size(0), i_emb.size(0)], dim=0)

            scores = torch.sum(u_g_embeddings * i_g_embeddings, dim=1)
            prediction = F.log_softmax(scores.unsqueeze(1), dim=1)
            predictions.append(prediction)

        return torch.cat(predictions, dim=0)


# NGCF 모델 정의
# class NGCF(nn.Module):
#     def __init__(self, num_users, num_items, emb_size, layers):
#         super(NGCF, self).__init__()
#         self.user_embedding = nn.Embedding(num_users, emb_size)
#         self.item_embedding = nn.Embedding(num_items, emb_size)
#         self.num_users = num_users  # num_users를 클래스의 속성으로 정의
#         self.num_items = num_items  # num_items를 클래스의 속성으로 정의
#         self.GC_layers = nn.ModuleList()
#         self.Bi_layers = nn.ModuleList()
#         for From, To in zip(layers[:-1], layers[1:]):
#             self.GC_layers.append(nn.Linear(From, To))
#             self.Bi_layers.append(nn.Linear(From, To))

#     def forward(self, user_indices, item_indices, adj_matrix):
#         u_emb = self.user_embedding(user_indices)
#         i_emb = self.item_embedding(item_indices)
#         ego_embeddings = torch.cat([u_emb, i_emb], 0)
#         all_embeddings = [ego_embeddings]
#         for k in range(len(self.GC_layers)):
#             side_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
#             sum_embeddings = F.leaky_relu(self.GC_layers[k](side_embeddings))
#             bi_embeddings = F.leaky_relu(self.Bi_layers[k](ego_embeddings * side_embeddings))
#             ego_embeddings = sum_embeddings + bi_embeddings
#             all_embeddings += [ego_embeddings]
#         all_embeddings = torch.cat(all_embeddings, 1)
#         u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
#         return u_g_embeddings, i_g_embeddings