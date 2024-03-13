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



# NGCF 모델 정의
class NGCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, layers):
        super(NGCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.GC_layers = nn.ModuleList()
        self.Bi_layers = nn.ModuleList()
        for From, To in zip(layers[:-1], layers[1:]):
            self.GC_layers.append(nn.Linear(From, To))
            self.Bi_layers.append(nn.Linear(From, To))

    def forward(self, user_indices, item_indices, adj_matrix):
        u_emb = self.user_embedding(user_indices)
        i_emb = self.item_embedding(item_indices)
        ego_embeddings = torch.cat([u_emb, i_emb], 0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.GC_layers)):
            side_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_layers[k](side_embeddings))
            bi_embeddings = F.leaky_relu(self.Bi_layers[k](ego_embeddings * side_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [num_users, num_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    
