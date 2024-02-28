# main.py
from dl_proc import DataSplitter  # 데이터 분할 및 로딩 클래스

import dl_proc as dp
import m2
import torch

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    dataset_path = "path/to/your/dataset"  # 데이터셋 경로
    model = m2.m2_model(num_node_features=3, num_classes=20774).to(device)  # `MyModel`은 사용자의 모델 클래스
    lr = 0.001  # 학습률

    # `gDataset` 인스턴스 생성
    dataset = dp.gDataset(dataset_path)

    # `Trainer` 인스턴스 생성
    trainer = m2.Trainer(model=model, dataset=dataset, lr=lr)

    # 훈련 시작
    trainer.train(epochs=500)
    
if __name__ == "__main__":
    main()

#m1 모델 loss 253~255 반복
#m2 모델 accuracy 32~33 