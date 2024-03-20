import torch
from torch_geometric.data import Dataset

from torch_geometric.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from ncsv2graph import GraphData
from torch_geometric.data import Data,Batch

from ncsv2graph import GraphData, WeightedAdjacencyMatrixCreator
import torch
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as GeoDataLoader

class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.GraphData = GraphData(dataset_path)
        self.GraphData.prepare_data()
        self.GraphData.create_individual_graphs()
        self.formakelabel = self.GraphData.graphs #네트워크 X 그래프
        self.GraphData.create_pyg_list()
        self.graph_data = self.GraphData.get_pyg_graphs() #pyg그래프
        self.adj_matrices_creator = WeightedAdjacencyMatrixCreator(self.GraphData) # 인스턴스 생성
        self.adj_matrices = self.adj_matrices_creator.process_all_graphs() # 인접 행렬 생성
        self.user_indices, self.item_indices = self.GraphData.get_user_item_indices() # 사용자 및 아이템 인덱스 가져오기
        self.item_labels = self.get_item_labels()


    def __len__(self):
        """
        데이터셋의 총 그래프 수를 반환합니다.
        """
        return len(self.graph_data)
    
    def get_item_labels(self):
        item_labels = []
        for graph in self.formakelabel:
            labels = []
            for node, attr in graph.nodes(data=True):
                if attr['type'] == 'item':
                    labels.append(attr['name'])
            item_labels.append(labels)
        return item_labels

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 그래프 데이터와 인접 행렬을 반환합니다.
        """
        graph = self.graph_data[idx]
        adj_matrix = self.adj_matrices[idx]
        labels = self.item_labels[idx]

        return {
            'user_indices': self.user_indices,
            'item_indices': self.item_indices,
            'graph': graph,
            'adj_matrix': adj_matrix,
            'labels': labels
        }

class NDataSplitter:
    def __init__(self, dataset_path="Dataset/최종합데이터.csv", batchsize=1, test_size=0.1, val_size=0.2):
        self.batch_size = batchsize
        self.test_size = test_size
        self.val_size = val_size

        # GraphData 인스턴스 생성
        graph_data = GraphData(dataset_path)
        self.dataset = CustomDataset(graph_data)
        
    def split_data(self):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        
        train_val_indices, test_indices = train_test_split(indices, test_size=self.test_size, random_state=42)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=self.val_size / (1 - self.test_size), random_state=42)
        
        # Subset을 사용하여 각 DataLoader에 대한 데이터셋을 지정합니다.
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)
        
        # PyTorch Geometric의 DataLoader를 사용합니다.
        train_loader = GeoDataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = GeoDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = GeoDataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
