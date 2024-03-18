import torch
from torch_geometric.data import Dataset
import ncsv2graph as nc2g 
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from ncsv2graph import GraphData


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.graph_data = GraphData(root)
        self.pyg_graphs = [self.graph_data.graph_to_pyg_data(g) for g in self.graph_data.graphs]
        
    def len(self):
        return len(self.pyg_graphs)

    def get(self, idx):
        return self.pyg_graphs[idx]

class NDataSplitter:
    def __init__(self, dataset_path="Dataset\최종합데이터.csv", batchsize=32, test_size=0.1, val_size=0.2):
        self.dataset_path = dataset_path
        self.batch_size = batchsize
        self.test_size = test_size
        self.val_size = val_size
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
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

