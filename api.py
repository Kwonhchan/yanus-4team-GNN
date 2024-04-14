import os
import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv
import io

# NGCF 모델 정의
class NGCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, layers, heads, num_classes):
        super(NGCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.emb_size = emb_size
        self.GC_layers = nn.ModuleList()
        self.GAT_layers = nn.ModuleList()
        self.GraphSAGE_layers = nn.ModuleList()
        self.Cheb_layers = nn.ModuleList()
        self.MLP_layers = nn.Sequential(
            # MLP 구조 정의
        )
        self.batch_norm_layers = nn.ModuleList()

        input_dim = 512
        for output_dim in layers:
            self.GC_layers.append(GCNConv(input_dim, output_dim))
            self.batch_norm_layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        self.GraphSAGE_layers.append(SAGEConv(input_dim, input_dim))
        self.Cheb_layers.append(ChebConv(input_dim, input_dim, K=2))

        for _ in range(heads):
            self.GAT_layers.append(GATConv(input_dim, input_dim, heads=1, concat=False))

        self.prediction_layer = nn.Linear(input_dim, num_classes)

    def forward(self, data):
        edge_index = data.edge_index
        
        user_indices = data.x[:, 0].long()
        item_indices = data.x[:, 1].long()

        u_emb = self.user_embedding(user_indices)
        i_emb = self.item_embedding(item_indices)

        x = torch.cat([u_emb, i_emb], dim=1)
        x = self.MLP_layers(x)
        
        for gc_layer, bn_layer in zip(self.GC_layers, self.batch_norm_layers):
            x = F.relu(bn_layer(gc_layer(x, edge_index)))
        x = F.relu(self.GraphSAGE_layers[0](x, edge_index))
        x = F.relu(self.Cheb_layers[0](x, edge_index))

        for gat_layer in self.GAT_layers:
            x = F.elu(gat_layer(x, edge_index))

        scores = self.prediction_layer(x)
        return scores

# GraphData 클래스 정의
class GraphData:
    def __init__(self, df):
        self.df = df
        self.prepare_data()
        self.graphs = []
        self.create_individual_graphs()
        self.pyg_graphs = []
        self.create_pyg_list()

    def prepare_data(self):
        # LabelEncoder를 사용하여 'TRAVEL_ID'와 'VISIT_AREA_NM'을 각각 인덱스로 변환
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        self.df['user_index'] = self.user_encoder.fit_transform(self.df['TRAVEL_ID'])
        self.df['item_index'] = self.item_encoder.fit_transform(self.df['VISIT_AREA_NM'])
        
        # 'GENDER' 칼럼을 숫자로 변환하여 새로운 칼럼 'GENDER_index' 생성
        self.df['GENDER_index'] = self.user_encoder.fit_transform(self.df['GENDER'])

        self.create_individual_graphs()

    def get_user_item_indices(self):
        graph_user_item_indices = []

        for G in self.graphs:
            # 각 그래프에서 사용자 노드와 아이템 노드의 인덱스를 추출합니다.
            user_indices = [node for node, attr in G.nodes(data=True) if attr['type'] == 'user']
            item_indices = [node for node, attr in G.nodes(data=True) if attr['type'] == 'item']

            graph_user_item_indices.append((user_indices, item_indices))

        return graph_user_item_indices

    def create_individual_graphs(self):
        # 'TRAVEL_ID' 별로 그룹화하여 각 그래프 생성
        for travel_id, group in self.df.groupby('TRAVEL_ID'):
            G = nx.Graph()
            user_index = group['user_index'].iloc[0]  # 모든 row에서 user_index는 동일합니다.
            
            # 사용자 노드에 속성 추가
            user_attributes = group.iloc[0][['GENDER_index', 'AGE_GRP', 'FAMILY_MEMB', 'TRAVEL_COMPANIONS_NUM']].to_dict()
            G.add_node(user_index, **user_attributes, type='user')

            for _, row in group.iterrows():
                item_index = row['item_index']
                # 아이템 노드 추가 (여기서는 아이템 노드에 별도의 속성을 지정하지 않음)
                G.add_node(item_index, type='item', name=row['VISIT_AREA_NM'])

                # 엣지 추가 및 엣지 속성 설정
                edge_attributes = row[['RESIDENCE_TIME_MIN', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']].to_dict()
                G.add_edge(user_index, item_index, **edge_attributes)
            
            self.graphs.append(G)

    def graph_to_pygdata(self, G):
        node_features = []
        node_labels = []
        node_index_mapping = {}
        for i, (node, attr) in enumerate(G.nodes(data=True)):
            node_index_mapping[node] = i
            if 'type' in attr and attr['type'] == 'user':
                node_features.append([attr.get('GENDER_index', 0), attr.get('AGE_GRP', 0),
                                    attr.get('FAMILY_MEMB', 0), attr.get('TRAVEL_COMPANIONS_NUM', 0)])
                node_labels.append(-1)
            else:
                node_features.append([0, 0, 0, 0])
                node_labels.append(self.item_encoder.transform([attr['name']])[0])

        edge_index = []
        edge_attr = []
        for source, target, attr in G.edges(data=True):
            edge_index.append([node_index_mapping[source], node_index_mapping[target]])
            edge_attr.append([attr.get('RESIDENCE_TIME_MIN', 0), attr.get('DGSTFN', 0),
                            attr.get('REVISIT_INTENTION', 0), attr.get('RCMDTN_INTENTION', 0)])
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_labels = torch.tensor(node_labels, dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=node_labels)
        
        return data

    def create_pyg_list(self):
        for G in self.graphs:
            pyg_data = self.graph_to_pygdata(G)
            self.pyg_graphs.append(pyg_data)

    def get_pyg_graphs(self):
        return self.pyg_graphs

# FastAPI 애플리케이션 정의
app = FastAPI()

# 모델 로드 및 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"model_checkpoint\best_model_epoch_4_val_loss_9.1074_val_acc_0.0044.pt"
df = pd.read_csv('Dataset/최종합데이터.csv')
unique_travel_ids_count = df['TRAVEL_ID'].nunique()
unique_VISIT_AREA_NM_ids_count = df['VISIT_AREA_NM'].nunique()
num_users, num_items = unique_travel_ids_count, unique_VISIT_AREA_NM_ids_count
# num_classes = 10844
num_classes = 41476
model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[128, 64, 32], heads=1, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

graph_data = GraphData(df)

# 추천 API 엔드포인트 정의
class RecommendationRequest(BaseModel):
    raw_data: str

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    # 웹에서 받은 데이터 예제를 DataFrame으로 변환
    df = pd.read_csv(io.StringIO(request.raw_data))

    # GraphData 객체 생성
    graph_data = GraphData(df)

    printed_recommendations = set()
    recommendations = []

    for data in graph_data.pyg_graphs:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        predicted_item_index = output.argmax(dim=1).cpu().numpy()
        predicted_item_names = graph_data.item_encoder.inverse_transform(predicted_item_index)

        # 중복 제거
        unique_recommendations = [item for item in predicted_item_names if item not in printed_recommendations]
        printed_recommendations.update(unique_recommendations)

        if unique_recommendations:
            recommendations.extend(unique_recommendations)

    return {"recommendations": recommendations}