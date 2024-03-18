import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nmodel import NGCF  # 모델 클래스 이름 확인 필요
from ncustom import CustomDataset, NDataSplitter  # CustomDataset 및 NDataSplitter 임포트

# CUDA 사용 가능 여부 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터셋 경로 설정 (경로 수정 필요)
data_path = 'Dataset/최종합데이터.csv'

# CustomDataset 인스턴스 생성
dataset = CustomDataset(root=data_path)

# DataLoader 생성 및 데이터 분할
data_splitter = NDataSplitter(dataset)
train_loader, val_loader, test_loader = data_splitter.split_data()

# 모델, 옵티마이저 초기화 및 CUDA device로 이동
data = pd.read_csv('Dataset\최종합데이터.csv')
num_users = data['TRAVEL_ID'].nunique()
num_items = data['VISIT_AREA_NM'].nunique()
model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[64, 64, 64]).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

# TensorBoard 설정
writer = SummaryWriter()

# 최고 정확도 및 체크포인트 초기화
best_accuracy = 0.0

# 학습 루프
for epoch in range(100):
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    total_train = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/100 Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item() * data.num_graphs
        preds = out.argmax(dim=1)
        total_train_correct += preds.eq(data.y).sum().item()
        total_train += data.num_graphs
    
    train_loss = total_train_loss / total_train
    train_acc = total_train_correct / total_train
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)

    # 검증 루프
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/100 Validation"):
            data = data.to(device)
            pred = model(data)
            loss = F.nll_loss(pred, data.y)
            total_val_loss += loss.item() * data.num_graphs
            preds = pred.argmax(dim=1)
            total_val_correct += preds.eq(data.y).sum().item()
            total_val += data.num_graphs
    
    val_loss = total_val_loss / total_val
    val_acc = total_val_correct / total_val
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Accuracy/Val', val_acc, epoch)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 모델 파일 이름에 성능 메트릭 포함
    model_filename = f"model_epoch{epoch+1}_trainLoss{train_loss:.4f}_trainAcc{train_acc:.4f}_valLoss{val_loss:.4f}_valAcc{val_acc:.4f}.pth"
    
    # 가장 좋은 모델 저장
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), model_filename)
        print(f"New best model saved as {model_filename} with Val Acc: {best_accuracy:.4f}")

# TensorBoard 로거 종료
writer.close()