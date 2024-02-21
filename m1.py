import torch
import torch.nn.functional as F
from tqdm import tqdm
import dl_proc as dl  # csv2graph_D를 포함하는 모듈
from torch_geometric.nn import GCNConv, GATConv

# GNN 모델 정의
class m1(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(m1, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GATConv(16, 32, heads=4, concat=True)
        self.conv3 = GCNConv(32 * 4, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


# 모델, 옵티마이저, 손실 함수 초기화
model = m1(num_features=3, num_classes=20774)  # num_classes는 문제의 요구에 따라 조정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 데이터셋 로드 및 데이터 로더 설정
dataset = dl.gDataset('Dataset/최종합데이터.csv')
loader = dl.gDataLoader(dataset, batch_size=32, shuffle=True)

# 모델 학습
model.train()
for epoch in range(100):
    total_loss = 0
    for data in tqdm(loader, desc=f'Epoch {epoch+1}'):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Total Loss: {total_loss}')
    
torch.save(model, 'model/prime_model.pth')
