import torch
import torch.utils.data
from torch_geometric.data.data import BaseData

from csv2graph import csv2graph_D
from torch_geometric.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch.utils.data import Subset

#CustomDataset
class gDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.csv2graph = csv2graph_D(dataset_path)
        self.csv2graph.convert_to_graph()
        self.csv2graph.convert_to_pyg_data()
        self.graphs = self.csv2graph.get_pyg_graphs()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        pyg_data = self.graphs[idx]
        return pyg_data


# class gDataLoader(DataLoader):
#     def __init__(self, *args, **kwargs):
#         super(gDataLoader, self).__init__(*args, **kwargs)

class DataSplitter:
    def __init__(self, dataset_path, batchsize=32, test_size=0.1, val_size=0.2):
        self.dataset_path = dataset_path
        self.batch_size = batchsize
        self.test_size = test_size
        self.val_size = val_size
        self.dataset = gDataset(dataset_path)
        
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