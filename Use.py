import torch
from torch_geometric.data import DataLoader

def load_model(model_path, model, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드로 설정
    return model

def generate_recommendations(model, data_loader, top_k, device):
    recommendations = {}
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)
        scores, indices = torch.topk(output, k=top_k, dim=1)  # 각 사용자별로 top-k 추천 아이템의 인덱스
        batch_user_ids = batch.user_ids.cpu().numpy()
        batch_item_indices = indices.cpu().numpy()
        for user_id, item_indices in zip(batch_user_ids, batch_item_indices):
            recommended_items = [data_loader.dataset.item_encoder.classes_[idx] for idx in item_indices]
            recommendations[user_id] = recommended_items
    return recommendations

# 모델 경로 설정 (학습한 가장 좋은 모델)
best_model_path = "model_checkpoint\best_model_epoch_4_val_loss_9.1074_val_acc_0.0044.pt"  # 예시 경로, 실제 경로로 대체해야 함
top_k = 5  # 사용자별로 추천할 아이템 수

# 모델 로딩
model = load_model(best_model_path, model, device)

# 테스트 데이터 로더 준비
# 여기서는 `test_loader`를 이미 생성했다고 가정합니다.
# 실제로는 테스트 데이터셋을 DataLoader로 변환하는 과정이 필요할 수 있습니다.

# 추천 생성
recommendations = generate_recommendations(model, test_loader, top_k, device)

# 추천 결과 출력
for user_id, items in recommendations.items():
    print(f"User {user_id}: Recommended items - {items}")
