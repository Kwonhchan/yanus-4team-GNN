import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from itertools import combinations


# path='Dataset\최종합데이터.csv'

import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

class csv2graph_D:
    def __init__(self, path):
        self.path = path
        self.graphs = {}  # 사용자별 그래프를 저장할 딕셔너리
        self.pyg_graphs = []  # PyTorch Geometric 그래프 데이터 저장

    # 라벨 인코딩
    def label_encoding(self):
        data = pd.read_csv(self.path)
        label_encoder = LabelEncoder()
        data['TRAVEL_ID'] = label_encoder.fit_transform(data['TRAVEL_ID'])
        data['VISIT_AREA_NM'] = label_encoder.fit_transform(data['VISIT_AREA_NM'])
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
            pyg_data = self.graph2torch(graph)
            self.pyg_graphs.append(pyg_data)

    # NetworkX 그래프를 PyTorch Geometric 데이터로 변환하는 내부 함수
    def graph2torch(self, graph):
        node_features = []
        edge_index = []
        edge_attr = []

        for node, data in graph.nodes(data=True):
            node_features.append(data['feature'])

        for source, target, data in graph.edges(data=True):
            edge_index.append([source, target])
            edge_attr.append([data['weight']])

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 생성된 PyTorch Geometric 그래프 데이터에 접근하는 메소드
    def get_pyg_graphs(self):
        return self.pyg_graphs