import torch
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecommendationSystem:
    def __init__(self, model, user_features, user_embeddings):
        """
        model: 학습된 GCN 모델
        user_features: 사용자 특성 정보가 담긴 텐서
        user_embeddings: 모델에 의해 생성된 사용자 임베딩
        """
        self.model = model
        self.user_features = user_features
        self.user_embeddings = user_embeddings

    def get_user_embedding(self, new_user_features):
        """
        새로운 사용자의 특성을 바탕으로 임베딩을 생성합니다.
        """
        self.model.eval()
        with torch.no_grad():
            new_user_embedding = self.model.user_embedding(new_user_features)
        return new_user_embedding

    def recommend_items(self, new_user_embedding, top_k=5):
        """
        새로운 사용자 임베딩을 바탕으로 가장 유사한 사용자를 찾고, 그들의 선호 아이템을 기반으로 추천합니다.
        """
        # 새로운 사용자와 기존 사용자 간의 유사도 계산
        similarities = cosine_similarity(new_user_embedding.numpy(), self.user_embeddings.numpy())
        top_k_users = np.argsort(similarities, axis=0)[-top_k:]
        
        # 여기에서는 유사한 사용자들의 선호 아이템을 찾고, 가장 인기 있는 아이템을 추천한다고 가정합니다.
        # 실제로는 top_k_users를 사용하여 해당 사용자들의 선호 아이템을 찾고, 가장 많이 선택된 아이템을 추천합니다.
        recommended_items = self.get_popular_items_among_users(top_k_users)
        
        return recommended_items

    def get_popular_items_among_users(self, user_indices):
        # 유사한 사용자들 사이에서 가장 인기 있는 아이템을 찾는 로직을 구현합니다.
        # 이 부분은 프로젝트의 구체적인 세부 정보에 따라 달라집니다.
        # 예시를 위해 임의의 아이템을 반환합니다.
        return ["item1", "item2", "item3", "item4", "item5"]