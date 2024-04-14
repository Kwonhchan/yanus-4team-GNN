import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as DataLoader
from torch_geometric.data import DataLoader as GeoDataLoader, Data
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset
import networkx as nx
from torch.nn import ModuleList, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv

from torch_geometric.data import Data
import numpy as np
import io
import pickle
from ncsv2graph import GraphData

# pickle 파일로부터 CustomDataset 객체 로드
with open('custom_dataset.pkl', 'rb') as f:
    loaded_custom_dataset = pickle.load(f)
    
class ExtendedLabelEncoder1:
    def __init__(self, base_encoder):
        # base_encoder는 LabelEncoder의 인스턴스입니다.
        self.base_encoder = base_encoder
    
    def fit(self, y):
        self.base_encoder.fit(y)
        return self

    def transform(self, y, unknown_label=-1):
        new_y = []
        for item in y:
            if item in self.base_encoder.classes_:
                # 기존 base_encoder의 transform 메서드를 사용하여 변환합니다.
                encoded = self.base_encoder.transform([item])[0]
                new_y.append(encoded)
            else:
                # 새로운 항목은 unknown_label 값으로 처리
                new_y.append(unknown_label)
        return np.array(new_y)
    
class ExtendedLabelEncoder2:
    def __init__(self, base_encoder):
        self.base_encoder = base_encoder  # 기존 LabelEncoder 인스턴스
        self.unknown_label_start = -1
        self.unknown_labels_dict = {}
    
    def fit(self, y):
        self.base_encoder.fit(y)
        return self

    def transform(self, y, unknown_label_start=-1):
        new_y = []
        for item in y:
            try:
                # base_encoder의 classes_를 참조하여 transform을 시도합니다.
                if item in self.base_encoder.classes_:
                    encoded = self.base_encoder.transform([item])[0]
                    new_y.append(encoded)
                else:
                    raise ValueError
            except ValueError:
                # 새로운 아이템 처리
                if item not in self.unknown_labels_dict:
                    self.unknown_labels_dict[item] = self.unknown_label_start
                    self.unknown_label_start -= 1
                new_y.append(self.unknown_labels_dict[item])
        return np.array(new_y)

# 모델 정의
class NGCF(torch.nn.Module):
    def __init__(self, num_users, num_items, emb_size, layers, heads, num_classes):
        super(NGCF, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, emb_size)
        self.item_embedding = torch.nn.Embedding(num_items, emb_size)
        self.emb_size = emb_size
        self.GC_layers = ModuleList()
        self.GAT_layers = ModuleList()
        self.GraphSAGE_layers = ModuleList()
        self.Cheb_layers = ModuleList()
        self.MLP_layers = Sequential(
            Linear(emb_size * 2, 1024),
            ReLU(),
            Dropout(0.5),
            Linear(1024, 512),
            ReLU(),
            Dropout(0.5)
        )
        self.batch_norm_layers = ModuleList()

        input_dim = 512  # MLP 후 출력 차원
        
        for output_dim in layers:
            self.GC_layers.append(GCNConv(input_dim, output_dim))
            self.batch_norm_layers.append(BatchNorm1d(output_dim))
            input_dim = output_dim

        self.GraphSAGE_layers.append(SAGEConv(input_dim, input_dim))
        self.Cheb_layers.append(ChebConv(input_dim, input_dim, K=2))

        for _ in range(heads):
            self.GAT_layers.append(GATConv(input_dim, input_dim, heads=1, concat=False))

        self.prediction_layer = Linear(input_dim, num_classes)

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

class GraphDataU:
    def __init__(self, df):
        self.df = df
        self.user_encoder = ExtendedLabelEncoder1(loaded_custom_dataset.user_encoder)
        self.item_encoder = ExtendedLabelEncoder2(loaded_custom_dataset.item_encoder)
        self.gender_encoder = ExtendedLabelEncoder1(loaded_custom_dataset.gender_encoder)
        self.prepare_data()
        self.graphs = []
        self.create_individual_graphs()
        self.pyg_graphs = []
        self.create_pyg_list()

    def prepare_data(self):
        self.df['user_index'] = self.user_encoder.transform(self.df['TRAVEL_ID'],unknown_label=-1)
        self.df['item_index'] = self.item_encoder.transform(self.df['VISIT_AREA_NM'])
        self.df['GENDER_index'] = self.gender_encoder.transform(self.df['GENDER'],unknown_label=-1)

    def create_individual_graphs(self):
        for _, group in self.df.groupby('TRAVEL_ID'):
            G = nx.Graph()
            user_index = group['user_index'].iloc[0]
            user_attributes = group.iloc[0][['GENDER_index', 'AGE_GRP', 'FAMILY_MEMB', 'TRAVEL_COMPANIONS_NUM']].to_dict()
            G.add_node(user_index, **user_attributes, type='user')
            for _, row in group.iterrows():
                item_index = row['item_index']
                G.add_node(item_index, type='item', name=row['VISIT_AREA_NM'])
                edge_attributes = row[['RESIDENCE_TIME_MIN', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']].to_dict()
                G.add_edge(user_index, item_index, **edge_attributes)
            self.graphs.append(G)

    def graph_to_pygdata(self, G):
        node_features, node_labels, edge_index, edge_attr = [], [], [], []
        node_index_mapping = {node: i for i, (node, _) in enumerate(G.nodes(data=True))}
        for node, attr in G.nodes(data=True):
            if 'type' in attr and attr['type'] == 'user':
                node_features.append([attr['GENDER_index'], attr['AGE_GRP'], attr['FAMILY_MEMB'], attr['TRAVEL_COMPANIONS_NUM']])
                node_labels.append(-1)
            else:
                node_features.append([0, 0, 0, 0])
                node_labels.append(self.item_encoder.transform([attr['name']])[0])
        for source, target, attr in G.edges(data=True):
            edge_index.append([node_index_mapping[source], node_index_mapping[target]])
            edge_attr.append([attr['RESIDENCE_TIME_MIN'], attr['DGSTFN'], attr['REVISIT_INTENTION'], attr['RCMDTN_INTENTION']])
        data = Data(x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                    y=torch.tensor(node_labels, dtype=torch.long))
        return data

    def create_pyg_list(self):
        for G in self.graphs:
            self.pyg_graphs.append(self.graph_to_pygdata(G))

# 웹에서 받은 데이터 예제
raw_data = """TRAVEL_ID,VISIT_AREA_NM,RESIDENCE_TIME_MIN,DGSTFN,REVISIT_INTENTION,RCMDTN_INTENTION,GENDER,AGE_GRP,FAMILY_MEMB,TRAVEL_COMPANIONS_NUM
d_d007782,감성점빵,30.0,4.0,3.0,3.0,여,20,1,1
d_d007782,국수먹고가게,30.0,4.0,2.0,2.0,여,20,1,1
d_d007782,이창명짜장면시키신분,30.0,3.0,2.0,2.0,여,20,1,1
"""
df1 = pd.read_csv(io.StringIO(raw_data))

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"model_checkpoint\best_model_epoch_1_val_loss_10.5523_val_acc_0.0009.pt" # 이전에 학습된 모델의 경로

df = pd.read_csv('Dataset/최종합데이터.csv')
unique_travel_ids_count = df['TRAVEL_ID'].nunique()
unique_VISIT_AREA_NM_ids_count = df['VISIT_AREA_NM'].nunique()
num_users, num_items = unique_travel_ids_count, unique_VISIT_AREA_NM_ids_count
num_classes = 41476
model = NGCF(num_users=num_users, num_items=num_items, emb_size=64, layers=[128, 64, 32], heads=1, num_classes=num_classes).to(device) # 파라미터는 예시입니다
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("변환중")
#사용자 데이터를 PyTorch Geometric 데이터로 변환
graph_data = GraphDataU(df1)
# path = 'Dataset/최종합데이터.csv'
# graph_data_T = GraphData(path)

# loaded_custom_dataset에서 graph_data 속성을 추출
# graph_data_T = loaded_custom_dataset.GraphData

print("변환완료")
recommendations = []  # 모든 추천 항목을 저장할 리스트

# 단일 사용자의 데이터 처리를 가정하여 첫 번째 그래프 데이터만 사용
data = graph_data.pyg_graphs[0].to(device)
with torch.no_grad():
    print("모델 돌아가는 중")
    output = model(data)
    print("결과")
predicted_item_index = output.argmax(dim=1).cpu().numpy()
predicted_item_names = loaded_custom_dataset.item_encoder.inverse_transform(predicted_item_index)

# 모든 추천 항목 출력 (중복 포함)
print(f"All recommended items for the user: {predicted_item_names}")
