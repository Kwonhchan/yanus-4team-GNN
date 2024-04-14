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
import pickle




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.GraphData = GraphData(dataset_path)
        self.GraphData.create_pyg_list()
        self.graph_data = self.GraphData.get_pyg_graphs()
        self.user_encoder = GraphData.user_encoder
        self.item_encoder = GraphData.item_encoder
        self.gender_encoder = GraphData.gender_encoder

    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        # PyG Data 객체 직접 반환
        return self.graph_data[idx]

class NDataSplitter:
    def __init__(self, dataset_path="Dataset/최종합데이터.csv", batchsize=4, test_size=0.1, val_size=0.2):
        self.batch_size = batchsize
        self.test_size = test_size
        self.val_size = val_size

        # GraphData 인스턴스 생성
        graph_data = GraphData(dataset_path)
        self.dataset = CustomDataset(dataset_path)
        
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


