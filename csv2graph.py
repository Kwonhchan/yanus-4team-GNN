import pandas as pd
import networkx as nx
import torch

from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from torch_geometric.data import Data


class csv2graph_D:
    def __init__(self, path):
        self.path = 'Dataset\최종합데이터.csv'
        self.graphs = {}  # 사용자별 그래프를 저장할 딕셔너리
        self.pyg_graphs = []  # PyTorch Geometric 그래프 데이터 저장
        self.visit_area_to_index = {}  # 방문 지역 인덱스 매핑

    # 라벨 인코딩 및 방문 지역 인덱스 매핑 생성
    def label_encoding(self):
        data = pd.read_csv(self.path)
        label_encoder = LabelEncoder()
        data['TRAVEL_ID'] = label_encoder.fit_transform(data['TRAVEL_ID'])
        
        # 방문 지역 인덱스 매핑 생성
        unique_visit_areas = data['VISIT_AREA_NM'].unique()
        self.visit_area_to_index = {area: idx for idx, area in enumerate(unique_visit_areas)}
        
        data['VISIT_AREA_NM'] = data['VISIT_AREA_NM'].map(self.visit_area_to_index)
        return data

    # 그래프 변환
    def convert_to_graph(self):
        data = self.label_encoding()
        for user_id, user_data in data.groupby('TRAVEL_ID'):
            user_graph = nx.Graph()
            for index, row in user_data.iterrows():
                node_id = row['VISIT_AREA_NM']
                feature = row[['RESIDENCE_TIME_MIN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']].tolist()
                user_graph.add_node(node_id, feature=feature)
            for source, target in combinations(user_graph.nodes, 2):
                weight = user_data.loc[(user_data['VISIT_AREA_NM'] == source) | (user_data['VISIT_AREA_NM'] == target), 'DGSTFN'].mean()
                user_graph.add_edge(source, target, weight=weight)
            self.graphs[user_id] = user_graph

    # PyTorch Geometric 데이터로 변환
    def convert_to_pyg_data(self):
        for user_id, graph in self.graphs.items():
            graph = self.reindex_nodes(graph)
            pyg_data = self.graph2torch(graph)
            self.pyg_graphs.append(pyg_data)

    # NetworkX 그래프를 PyTorch Geometric 데이터로 변환하는 내부 함수
    def graph2torch(self, graph):
        node_features = []
        edge_index = []
        edge_attr = []
        y = []  # y 데이터를 저장할 리스트

        for node, data in graph.nodes(data=True):
            node_features.append(data['feature'])
            y.append(node)  # 노드 ID(방문 지역 인덱스)를 y 데이터로 사용

        for source, target, data in graph.edges(data=True):
            edge_index.append([source, target])
            edge_attr.append([data['weight']])

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)  # y 데이터를 텐서로 변환

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # 노드 재인덱싱 메소드 추가
    def reindex_nodes(self, graph):
        node_labels = list(graph.nodes())
        label_encoder = LabelEncoder()
        node_indices = label_encoder.fit_transform(node_labels)
        mapping = {old_label: new_index for old_label, new_index in zip(node_labels, node_indices)}
        return nx.relabel_nodes(graph, mapping)

    # PyTorch Geometric 그래프 데이터 리스트를 반환하는 메소드
    def get_pyg_graphs(self):
        return self.pyg_graphs
