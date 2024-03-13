import torch
import torch.nn.functional as F
import os
import dl_proc as dp

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TopKPooling, global_mean_pool

# GCN, GAT, GraphSAGE, Unet구조, skip-connection, residual Connections, 멀티-헤드 Attention 메커니즘
class m2_model(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(m2_model, self).__init__()
        # Down-sampling Path 제거됨
        self.sage1 = SAGEConv(num_node_features, 64)
        self.sage2 = SAGEConv(64, 128)

        # Bottleneck with multi-head attention
        self.gat = GATConv(128, 128, heads=8, concat=True)

        # Up-sampling Path
        self.gcn1 = GCNConv(128 * 8, 64)  # Adjusted for concatenated multi-head attention output
        self.gcn2 = GCNConv(64, num_classes)

        # Residual Connections and Dimension Matching
        self.res1 = torch.nn.Linear(num_node_features, 64)
        self.res2 = torch.nn.Linear(64, 128 * 8)  # Adjust for concatenated multi-head attention output
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial residual connections
        res_x = self.res1(x)

        # Contracting path without TopKPooling
        x = F.relu(self.sage1(x, edge_index)) + res_x
        x1 = x  # Skip connection

        x = F.relu(self.sage2(x, edge_index))

        # Bottleneck with GAT for attention mechanism
        x = F.relu(self.gat(x, edge_index))

        # Up-sampling path with GCN for refining features
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)

        return F.log_softmax(x, dim=1)

class Trainer:
    def __init__(self, model, dataset, lr=0.001):
        # CUDA 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # DataSplitter 인스턴스 생성 및 데이터 분할
        self.data_splitter = dp.DataSplitter(self.dataset)
        self.train_loader, self.val_loader, self.test_loader = self.data_splitter.split_data()

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(iter(self.train_loader), desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, data in enumerate(progress_bar):
                data = data.to(self.device)  # 전체 Data 객체를 디바이스로 이동

                # Forward pass
                outputs = self.model(data)

                # `data.y`는 레이블을 포함하고 있을 것으로 예상됩니다.
                loss = self.criterion(outputs, data.y)

                # Backward pass 및 최적화
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss/(batch_idx+1))
            
            self.validate()

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(), tqdm(self.val_loader, desc='Validating') as progress_bar:
            for data in progress_bar:
                data = data.to(self.device)  # Data 객체를 디바이스로 이동

                outputs = self.model(data)
                loss = self.criterion(outputs, data.y)  # 레이블은 data.y에 있다고 가정
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  # outputs가 이미 device에 있음
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()

                # 진행 상황 업데이트
                progress_bar.set_postfix(val_loss=val_loss/(progress_bar.last_print_n+1), accuracy=100. * correct / total)

        self.save_checkpoint(val_loss, 'model_checkpoint.pth')

    def save_checkpoint(self, val_loss, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
        }, filename)