import pandas as pd
import networkx as nx
import torch
import random

from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from torch_geometric.data import Data

class GraphData:
    def __init__(self, path):
        self.path = path
        self.graph = nx.Graph()
        self.graphs = []  # 그래프 리스트를 저장할 멤버 변수
        self.load_data()
        self.create_individual_graphs()

    def load_data(self):
        self.df = pd.read_csv(self.path)  # 파일 경로 수정
        return self.df
    
    def create_individual_graphs(self):  # 'self' 파라미터 추가
        """각 TRAVEL_ID 별로 별도의 그래프 데이터 생성하고 리스트로 반환"""
        graphs = []
        for travel_id, group in self.df.groupby('TRAVEL_ID'):  # self.df로 변경
            G = nx.Graph()
            first_row = group.iloc[0]
            G.add_node(travel_id, type='travel_id', gender=first_row['GENDER'], age_grp=first_row['AGE_GRP'], 
                    family_memb=first_row['FAMILY_MEMB'], travel_companions_num=first_row['TRAVEL_COMPANIONS_NUM'])
            for _, row in group.iterrows():
                G.add_node(row['VISIT_AREA_NM'], type='visit_area_nm')
                G.add_edge(travel_id, row['VISIT_AREA_NM'], 
                        residence_time_min=row['RESIDENCE_TIME_MIN'], 
                        dgstfn=row['DGSTFN'], 
                        revisit_intention=row['REVISIT_INTENTION'], 
                        rcmdtn_intention=row['RCMDTN_INTENTION'])
            graphs.append(G)
        self.graphs = graphs  # 생성된 그래프 리스트를 클래스 변수로 저장
    
    def split_data(self, train_ratio=0.8):
        """전체 그래프 데이터를 학습 및 검증 데이터셋으로 분할"""
        random.shuffle(self.graphs)  # 그래프 리스트를 무작위로 섞음
        split_idx = int(len(self.graphs) * train_ratio)
        return self.graphs[:split_idx], self.graphs[split_idx:]
    
    def convert_graphs_to_pyg(self):
        """NetworkX 그래프를 PyTorch Geometric Data 객체로 변환"""
        from torch_geometric.utils import from_networkx  # 필요한 함수 추가
        pyg_data_list = []
        for G in self.graphs:
            pyg_data = from_networkx(G)  # 수정된 변환 방법
            pyg_data_list.append(pyg_data)
        return pyg_data_list