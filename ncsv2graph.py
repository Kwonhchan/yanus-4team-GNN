import pandas as pd
import networkx as nx
import torch
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from torch_geometric.data import Data
import scipy.sparse as sp
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data

class GraphData:
    def __init__(self,path):
        self.path = 'Dataset\최종합데이터.csv'  # 경로를 생성자의 매개변수로 받아옵니다.
        self.df = pd.read_csv(self.path)
        self.graphs = []  # 그래프 리스트를 저장할 멤버 변수
        self.prepare_data()
        self.pyg_graphs = []

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



class WeightedAdjacencyMatrixCreator:
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.GraphData = GraphData(dataset_path)
        self.GraphData.prepare_data()
        self.GraphData.create_individual_graphs()
        # self.GraphData.graph_to_pygdata()
        self.GraphData.create_pyg_list()
        self.graph_data = self.GraphData.get_pyg_graphs()
        

    def calculate_edge_weight(self, edge_attr):
        # 체류 시간을 최소 0분, 최대 600분으로 가정하고 정규화합니다.
        residence_time = edge_attr[0]  # 체류 시간은 edge_attr의 첫 번째 요소
        normalized_residence_time = torch.clamp(residence_time, max=600) / 600
        
        # 만족도, 재방문 의도, 추천 의도의 평균을 계산합니다.
        positive_experiences_avg = (edge_attr[1] + edge_attr[2] + edge_attr[3]) / 3

        # 최종 가중치를 계산합니다.
        weight = normalized_residence_time + positive_experiences_avg
        
        return weight

    def create_adjacency_matrix(self, data):
        num_nodes = data.num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

        for src, dest in data.edge_index.t().tolist():
            edge_attr = data.edge_attr[src]
            weight = self.calculate_edge_weight(edge_attr)
            if src < num_nodes and dest < num_nodes:  # 인덱스가 유효한지 확인
                adj_matrix[src, dest] = weight  # 방향은 TRAVEL_ID 노드에서 VISIT_AREA_NM 노드로 설정합니다.
        return adj_matrix

    def process_all_graphs(self):
        adjacency_matrices = []
        for data in self.graph_data:  # 수정된 부분
            adj_matrix = self.create_adjacency_matrix(data)
            adjacency_matrices.append(adj_matrix)
        return adjacency_matrices