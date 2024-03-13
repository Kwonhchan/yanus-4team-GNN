import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nmodel import NGCF  # 모델 클래스 이름 수정 필요
from ncustom import CustomDataset  # CustomDataset 임포트
from sklearn.metrics import accuracy_score



# 데이터셋 경로 설정
data_path = 'Dataset\최종합데이터.csv'

# CustomDataset 인스턴스 생성
train_dataset = CustomDataset(root=data_path, dataset_type='train')
val_dataset = CustomDataset(root=data_path, dataset_type='val')

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델, 옵티마이저 초기화
model = NGCF()
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
    model_filename = f"model_epoch{epoch+1}_trainLoss{train_loss:.4f}_trainAcc{train_acc:.4f}_valLoss{val_loss:.4f}.pth"
    
    # 가장 좋은 모델 저장
    # 예시에서는 val_acc 기준으로 최고 모델을 저장합니다. 필요에 따라 다른 기준으로 변경 가능합니다.
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), model_filename)
        print(f"New best model saved as {model_filename} with Val Acc: {best_accuracy:.4f}")

# TensorBoard 로거 종료
writer.close()