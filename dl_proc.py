import torch
import torch.utils.data
from torch_geometric.data.data import BaseData

from csv2graph import csv2graph_D
from torch_geometric.data import Dataset, DataLoader

#CustomDataset
class gDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.csv2graph = csv2graph_D(path)
        self.csv2graph.convert_to_graph()
        self.csv2graph.convert_to_pyg_data()
        self.graphs = self.csv2graph.get_pyg_graphs()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class gDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(gDataLoader, self).__init__(*args, **kwargs)
