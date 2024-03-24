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
        self.num_users = num_users
        self.num_items = num_items
        self.GC_layers = nn.ModuleList()
        self.Bi_layers = nn.ModuleList()
        for From, To in zip(layers[:-1], layers[1:]):
            self.GC_layers.append(nn.Linear(From, To))
            self.Bi_layers.append(nn.Linear(From, To))

    def forward(self, user_indices, item_indices, adj_matrices):
        # adj_matrices는 (batch_size, num_nodes, num_nodes) 형태로 가정합니다.
        batch_size = adj_matrices.size(0)
        predictions = []

        for b in range(batch_size):
            # 현재 배치의 adj_matrix를 추출합니다.
            adj_matrix = adj_matrices[b]

            # 사용자 및 아이템의 초기 임베딩을 동적으로 추출합니다.
            u_emb = self.user_embedding(user_indices[b])
            i_emb = self.item_embedding(item_indices[b])

            # 초기 임베딩을 결합하여 전체 임베딩 행렬을 생성합니다.
            ego_embeddings = torch.cat([u_emb, i_emb], dim=0)

            all_embeddings = [ego_embeddings]

            # 그래프 컨볼루션 레이어를 통해 임베딩을 갱신합니다.
            for gc_layer, bi_layer in zip(self.GC_layers, self.Bi_layers):
                side_embeddings = torch.matmul(adj_matrix.to_dense(), ego_embeddings)
                sum_embeddings = F.leaky_relu(gc_layer(side_embeddings))
                bi_embeddings = F.leaky_relu(bi_layer(ego_embeddings * side_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                all_embeddings.append(ego_embeddings)

            all_embeddings = torch.cat(all_embeddings, dim=1)

            # 사용자와 아이템 임베딩을 분리합니다.
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [u_emb.size(0), i_emb.size(0)], dim=0)

            # 사용자와 아이템 임베딩의 내적을 통해 선호도 점수를 계산합니다.
            scores = torch.matmul(u_g_embeddings, i_g_embeddings.t())

            # 사용자가 관심을 가질 것으로 예상되는 아이템에 대한 확률을 계산합니다.
            prediction = F.log_softmax(scores, dim=1)
            predictions.append(prediction)

        # 배치별 예측 결과를 하나의 텐서로 결합합니다.
        predictions = torch.stack(predictions, dim=0)

        return predictions




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