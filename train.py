import torch
import torch.nn.functional as F
import pandas as pd

from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from nmodel import NGCF  # NGCF 모델 정의가 포함된 파일을 임포트
from ncsv2graph import GraphData, WeightedAdjacencyMatrixCreator  # 데이터 처리를 위한 클래스 임포트
from ncustom import CustomDataset, NDataSplitter  # 데이터셋 클래스와 데이터 분할 클래스 임포트

# # 데이터셋 경로
# dataset_path = "Dataset/최종합데이터.csv"

# data = pd.read_csv(dataset_path)
# user_encoder = LabelEncoder()
# item_encoder = LabelEncoder()
# data['user_index'] = user_encoder.fit_transform(data['TRAVEL_ID'])
# data['item_index'] = item_encoder.fit_transform(data['VISIT_AREA_NM'])

# # 사용자와 아이템의 총 수 계산
# num_users = data['user_index'].nunique()
# num_items = data['item_index'].nunique()

# # 학습 파라미터 설정
# num_users = data['user_index'].nunique()
# num_items = data['item_index'].nunique()
# emb_size = 64  # 임베딩 크기
# layers = [emb_size, 128, 64]  # 각 레이어의 출력 크기
# learning_rate = 0.001
# epochs = 20

# # 모델 및 옵티마이저 초기화
# model = NGCF(num_users, num_items, emb_size, layers)
# optimizer = Adam(model.parameters(), lr=learning_rate)

# # 손실 함수로 Cross Entropy Loss를 사용합니다.
# loss_function = torch.nn.CrossEntropyLoss()

# # 데이터 로더 준비
# splitter = NDataSplitter(dataset_path=dataset_path, batchsize=32, test_size=0.1, val_size=0.2)
# train_loader, val_loader, test_loader = splitter.split_data()

