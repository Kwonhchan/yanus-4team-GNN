import pandas as pd
import networkx as nx
import torch
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from torch_geometric.data import Data

class GraphData:
    def __init__(self, path):
        self.path = "Dataset\최종합데이터.csv"
        self.graph = nx.Graph()
        self.graphs = []  # 그래프 리스트를 저장할 멤버 변수
        self.load_data()
        self.create_individual_graphs()

    def load_data(self):
        self.df = pd.read_csv(self.path)  # 파일 경로 수정
        # VISIT_AREA_NM의 라벨 인코딩
        self.visit_area_encoder = LabelEncoder()
        self.df['VISIT_AREA_NM_encoded'] = self.visit_area_encoder.fit_transform(self.df['VISIT_AREA_NM'])
        return self.df

    def create_individual_graphs(self):
        """각 TRAVEL_ID 별로 별도의 그래프 데이터 생성하고 리스트로 반환, y 값 설정"""
        for travel_id, group in self.df.groupby('TRAVEL_ID'):
            G = nx.Graph()
            G.graph['name'] = travel_id  # 그래프 이름 설정
            # 가장 많이 방문된 방문지 결정
            most_visited_area = group['VISIT_AREA_NM_encoded'].value_counts().idxmax()
            G.graph['label'] = most_visited_area  # 그래프 라벨 설정
            # 첫 번째 행을 사용하여 TRAVEL_ID 노드의 특성을 설정
            first_row = group.iloc[0]
            G.add_node(travel_id, type='travel_id', gender=first_row['GENDER'], age_grp=first_row['AGE_GRP'],
                        family_memb=first_row['FAMILY_MEMB'], travel_companions_num=first_row['TRAVEL_COMPANIONS_NUM'])
            for _, row in group.iterrows():
                # VISIT_AREA_NM을 노드로 추가하고 TRAVEL_ID 노드와 연결, 엣지에 가중치 적용
                G.add_node(row['VISIT_AREA_NM'], type='visit_area_nm')
                G.add_edge(travel_id, row['VISIT_AREA_NM'],
                            residence_time_min=row['RESIDENCE_TIME_MIN'],
                            dgstfn=row['DGSTFN'],
                            revisit_intention=row['REVISIT_INTENTION'],
                            rcmdtn_intention=row['RCMDTN_INTENTION'])
            self.graphs.append(G)
        return self.graphs
    
    def graph_to_pyg_data(self, G):
        # 노드 ID와 노드 특성을 인코딩하기 위한 LabelEncoder 생성
        le = LabelEncoder()
        node_labels = le.fit_transform(list(G.nodes()))
        node_idx_mapping = {label: idx for idx, label in enumerate(G.nodes())}
        
        # 엣지 인덱스 생성
        edge_index = [[node_idx_mapping[s], node_idx_mapping[t]] for s, t in G.edges()]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 노드 특성 행렬 구성
        x_features = []
        for node, data in G.nodes(data=True):
            if data['type'] == 'travel_id':
                gender_encoded = 1 if data['gender'] == '남' else 0  # 성별을 숫자로 인코딩: 남성은 1, 여성은 0
                features = [
                    gender_encoded,
                    float(data['age_grp']),  # 연령대
                    float(data['family_memb']),  # 가족 구성원 수
                    float(data['travel_companions_num'])  # 동행자 수
                ]
            else:
                features = [0, 0, 0, 0]  # VISIT_AREA_NM 노드의 경우 특성을 0으로 설정
            x_features.append(features)
        x = torch.tensor(x_features, dtype=torch.float)
        
        # 엣지 특성 행렬 구성
        edge_attributes = []
        for s, t, data in G.edges(data=True):
            edge_attr = [
                float(data['residence_time_min']),
                float(data['dgstfn']),
                float(data['revisit_intention']),
                float(data['rcmdtn_intention'])
            ]
            edge_attributes.append(edge_attr)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        
        
        # PyG Data 객체 생성
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_label = G.graph['label']
        data.y = torch.tensor([graph_label], dtype=torch.long)
        
        return data