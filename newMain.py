import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader as GeoDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import os
import pickle
from nmodel import NGCF  # NGCF 모델 클래스
from ncsv2graph import GraphData  # GraphData 클래스
from ncustom import CustomDataset, NDataSplitter  # CustomDataset 및 NDataSplitter 클래스


def save_dataset_and_splitter(custom_dataset, data_splitter):
    with open('custom_dataset.pkl', 'wb') as f:
        pickle.dump(custom_dataset, f)
    with open('data_splitter.pkl', 'wb') as f:
        pickle.dump(data_splitter, f)
        
def load_dataset_and_splitter():
    with open('custom_dataset.pkl', 'rb') as f:
        custom_dataset = pickle.load(f)
    with open('data_splitter.pkl', 'rb') as f:
        data_splitter = pickle.load(f)
    return custom_dataset, data_splitter

class CustomLoss(nn.Module):
    def __init__(self, base_loss_function=nn.MSELoss()):
        super().__init__()
        self.base_loss_function = base_loss_function

    def forward(self, predictions, targets):
        # -1 레이블을 가진 타겟은 손실 계산에서 제외
        valid_indices = targets != -1
        if valid_indices.any():
            return self.base_loss_function(predictions[valid_indices], targets[valid_indices])
        else:
            return torch.tensor(0.0).to(predictions.device)  # 모든 타겟이 -1인 경우 0 반환


try:
    # 저장된 객체를 불러옵니다.
    custom_dataset, data_splitter = load_dataset_and_splitter()
    print("Loaded dataset and splitter from saved files.")
except (FileNotFoundError, IOError):
    print("Saved files not found. Creating new instances.")
    # 데이터셋과 데이터 스플리터 인스턴스를 새로 생성합니다.
    dataset_path = 'Dataset/최종합데이터.csv'
    custom_dataset = CustomDataset(dataset_path)
    data_splitter = NDataSplitter(dataset_path)

    # 객체를 저장합니다.
    save_dataset_and_splitter(custom_dataset, data_splitter)




# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('Dataset/최종합데이터.csv')
unique_travel_ids_count = df['TRAVEL_ID'].nunique()
unique_VISIT_AREA_NM_ids_count = df['VISIT_AREA_NM'].nunique()

num_users, num_items = unique_travel_ids_count, unique_VISIT_AREA_NM_ids_count
model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[128, 64, 32], heads=2).to(device)
# model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[128, 64, 32]).to(device)
optimizer = Adam(model.parameters(), lr=0.01)

criterion = CustomLoss()
# criterion = nn.MSELoss()  # 상호작용 예측을 위해 MSE Loss 사용

# 데이터 로더 생성 및 데이터 분할
train_loader, val_loader, test_loader = data_splitter.split_data()

# 학습 및 검증 루프
num_epochs = 2000


# TensorBoard 설정
writer = SummaryWriter()

#회귀 Loss 초기화
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch.y.float())  # 여기서 batch.y.float()으로 변경
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

    # 검증 단계
    model.eval()  # 모델을 평가 모드로 설정
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y.float())
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # TensorBoard에 학습 및 검증 손실 로깅
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)

    # 체크포인트 저장
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_path = f"bm_regression/best_model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved: {best_model_path} with Validation Loss: {val_loss:.4f}")

# TensorBoard 로거 종료
writer.close()

print("Training completed.")


# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from sklearn.preprocessing import LabelEncoder
# from torch_geometric.utils import to_dense_adj
# from torch_geometric.data import DataLoader
# import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from nmodel import NGCF  # 모델 클래스 이름 확인 필요
# from ncustom import CustomDataset, NDataSplitter  # CustomDataset 및 NDataSplitter 임포트
# from torch_geometric.utils import to_dense_adj

# # 데이터셋 경로 설정
# data_path = 'Dataset/최종합데이터.csv'

# # DataLoader 생성 및 데이터 분할
# dataset = CustomDataset(data_path)
# data_splitter = NDataSplitter(data_path)  # NDataSplitter 인스턴스 생성
# train_loader, val_loader, test_loader = data_splitter.split_data()

# # 모델 초기화 및 device 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # 사용자(User)와 아이템(Item) 인덱스 생성
# data = pd.read_csv(data_path)
# user_encoder = LabelEncoder()
# item_encoder = LabelEncoder()
# data['user_index'] = user_encoder.fit_transform(data['TRAVEL_ID'])
# data['item_index'] = item_encoder.fit_transform(data['VISIT_AREA_NM'])

# # 사용자와 아이템의 총 수 계산
# num_users = data['user_index'].nunique()
# num_items = data['item_index'].nunique()

# model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[64, 64, 64]).to(device)
# optimizer = Adam(model.parameters(), lr=0.001)

# # TensorBoard 설정
# writer = SummaryWriter()

# # 최고 정확도 및 체크포인트 초기화
# best_accuracy = 0.0

# # 학습 루프
# for epoch in range(100):
#     model.train()
#     total_train_loss = 0
#     total_train_correct = 0
#     total_train = 0
#     for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/100 Training"):
#         data = data.to(device)
#         item_indices = torch.arange(data.num_nodes, device=device)
#         adj_matrix = to_dense_adj(data.edge_index)[0]
#         optimizer.zero_grad()
#         out = model(item_indices=item_indices, adj_matrix=adj_matrix)
#         loss = F.nll_loss(out, data.y)
#         loss.backward()
#         optimizer.step()
        
#         total_train_loss += loss.item() * data.num_graphs
#         preds = out.argmax(dim=1)
#         total_train_correct += preds.eq(data.y).sum().item()
#         total_train += data.num_graphs
    
#     train_loss = total_train_loss / total_train
#     train_acc = total_train_correct / total_train
#     writer.add_scalar('Loss/Train', train_loss, epoch)
#     writer.add_scalar('Accuracy/Train', train_acc, epoch)

#     # 검증 루프
#     model.eval()
#     total_val_loss = 0
#     total_val_correct = 0
#     total_val = 0
#     with torch.no_grad():
#         for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/100 Validation"):
#             data = data.to(device)
#             pred = model(data)
#             loss = F.nll_loss(pred, data.y)
#             total_val_loss += loss.item() * data.num_graphs
#             preds = pred.argmax(dim=1)
#             total_val_correct += preds.eq(data.y).sum().item()
#             total_val += data.num_graphs
    
#     val_loss = total_val_loss / total_val
#     val_acc = total_val_correct / total_val
#     writer.add_scalar('Loss/Val', val_loss, epoch)
#     writer.add_scalar('Accuracy/Val', val_acc, epoch)

#     print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

#     # 모델 파일 이름에 성능 메트릭 포함
#     model_filename = f"model_epoch{epoch+1}_trainLoss{train_loss:.4f}_trainAcc{train_acc:.4f}_valLoss{val_loss:.4f}_valAcc{val_acc:.4f}.pth"
    
#     # 가장 좋은 모델 저장
#     if val_acc > best_accuracy:
#         best_accuracy = val_acc
#         torch.save(model.state_dict(), model_filename)
#         print(f"New best model saved as {model_filename} with Val Acc: {best_accuracy:.4f}")

# # TensorBoard 로거 종료
# writer.close()