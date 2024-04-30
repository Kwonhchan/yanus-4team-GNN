import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv

import torch.nn.functional as F

class NGCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, layers, heads, num_classes):
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
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.batch_norm_layers = nn.ModuleList()

        input_dim = 512  # MLP 후 출력 차원
        
        for output_dim in layers:
            self.GC_layers.append(GCNConv(input_dim, output_dim))
            self.batch_norm_layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        self.GraphSAGE_layers.append(SAGEConv(input_dim, input_dim))
        self.Cheb_layers.append(ChebConv(input_dim, input_dim, K=2))

        for _ in range(heads):
            self.GAT_layers.append(GATConv(input_dim, input_dim, heads=1, concat=False))

        self.prediction_layer = nn.Linear(input_dim, num_classes)

    def forward(self, data):
        edge_index = data.edge_index
        
        user_indices = data.x[:, 0].long()
        item_indices = data.x[:, 1].long()

        u_emb = self.user_embedding(user_indices)
        i_emb = self.item_embedding(item_indices)

        x = torch.cat([u_emb, i_emb], dim=1)
        x = self.MLP_layers(x)
        
        for gc_layer, bn_layer in zip(self.GC_layers, self.batch_norm_layers):
            x = F.relu(bn_layer(gc_layer(x, edge_index)))
        x = F.relu(self.GraphSAGE_layers[0](x, edge_index))
        x = F.relu(self.Cheb_layers[0](x, edge_index))

        for gat_layer in self.GAT_layers:
            x = F.elu(gat_layer(x, edge_index))

        scores = self.prediction_layer(x)

        # softmax를 통해 확률 분포로 변환
        probabilities = F.softmax(scores, dim=1)

        return probabilities