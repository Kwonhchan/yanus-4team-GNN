import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader as GeoDataLoader
from tqdm import tqdm
import pandas as pd
import os
import pickle
from nmodel_classifier import NGCF  # NGCF 모델 클래스
from ncsv2graph import GraphData  # GraphData 클래스
from ncustom import CustomDataset, NDataSplitter  # CustomDataset 및 NDataSplitter 클래스
import torch
import pandas as pd
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.optim as optim
from torch.optim import SparseAdam

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

def accuracy(output, target):
    valid_indices = target != -1
    if valid_indices.any():
        preds = output[valid_indices].argmax(dim=1)
        correct = (preds == target[valid_indices]).float()
        acc = correct.sum() / len(correct)
        return acc
    else:
        return 0.0

class CustomLoss(nn.Module):
    def __init__(self, base_loss_function=nn.CrossEntropyLoss()):
        super().__init__()
        self.base_loss_function = base_loss_function

    def forward(self, predictions, targets):
        # -1 레이블을 가진 타겟은 손실 계산에서 제외
        valid_indices = targets != -1
        if valid_indices.any():
            return self.base_loss_function(predictions[valid_indices], targets[valid_indices])
        else:
            return torch.tensor(0.0).to(predictions.device)  # 모든 타겟이 -1인 경우 0 반환



# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('Dataset/최종합데이터.csv')
unique_travel_ids_count = df['TRAVEL_ID'].nunique()
unique_VISIT_AREA_NM_ids_count = df['VISIT_AREA_NM'].nunique()

num_users, num_items = unique_travel_ids_count, unique_VISIT_AREA_NM_ids_count

num_classes = 41476
custom_loss_function = CustomLoss()
model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[128, 64, 32], heads=1, num_classes=num_classes).to(device)
# model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[128, 64, 32]).to(device)
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()

# 데이터 로더 생성 및 데이터 분할
train_loader, val_loader, test_loader = data_splitter.split_data()

# 학습 및 검증 루프
num_epochs = 10

# TensorBoard 설정
writer = SummaryWriter()

# 최적의 손실 초기화
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = custom_loss_function(predictions, batch.y)
        acc = accuracy(predictions, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
    
    train_loss = total_loss / len(train_loader)
    train_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

    # 검증 단계
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
            batch = batch.to(device)
            predictions = model(batch)
            loss = custom_loss_function(predictions, batch.y)
            acc = accuracy(predictions, batch.y)
            val_loss += loss.item()
            val_acc += acc.item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_acc / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # TensorBoard에 로깅
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_acc, epoch)

    # 체크포인트 저장
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_path = f"model_checkpoint/best_model_epoch_{epoch+1}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"New best modelsaved: {best_model_path} with Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
writer.close()

print("Training completed.")