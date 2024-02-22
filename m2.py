import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TopKPooling, global_mean_pool

# GCN, GAT, GraphSAGE, Unet구조, skip-connection, residual Connections, 멀티-헤드 Attention 메커니즘, TopKPooling
class m2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(m2, self).__init__()
        # Down-sampling Path
        self.sage1 = SAGEConv(num_node_features, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.sage2 = SAGEConv(64, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)

        # Bottleneck with multi-head attention
        self.gat = GATConv(128, 128, heads=8, concat=True)

        # Up-sampling Path
        self.gcn1 = GCNConv(128 * 8, 64)  # Adjusted for concatenated multi-head attention output
        self.gcn2 = GCNConv(64, num_classes)

        # Residual Connections
        self.res1 = torch.nn.Linear(num_node_features, 64)
        self.res2 = torch.nn.Linear(64, 128 * 8)  # Adjusted for concatenated multi-head attention output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial residual connections
        res_x = self.res1(x)

        # Contracting path with GraphSAGE
        x = F.relu(self.sage1(x, edge_index)) + res_x
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = x  # Skip connection

        # Prepare for the next residual connection
        res_x = self.res2(x)

        x = F.relu(self.sage2(x, edge_index)) + res_x
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        # Bottleneck with GAT for attention mechanism
        x = F.relu(self.gat(x, edge_index))

        # Expanding path with GCN for refining features
        x = F.relu(self.gcn1(x + x1, edge_index))  # Using skip connection
        x = self.gcn2(x, edge_index)

        # Global mean pooling
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)
    
    
