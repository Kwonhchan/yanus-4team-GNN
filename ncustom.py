import torch
from torch_geometric.data import Dataset
import ncsv2graph as nc2g 

class CustomDataset(Dataset):
    def __init__(self, root, dataset_type='train', transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.graph_data = nc2g.GraphData(root)

        # 여기서는 split_data 호출 후 self.graph_data.graphs를 업데이트하지 않고,
        # 직접 self.data_list를 설정합니다.
        train_data, val_data = self.graph_data.split_data()

        if dataset_type == 'train':
            self.graph_data.graphs = train_data
        elif dataset_type == 'val':
            self.graph_data.graphs = val_data
        else:
            raise ValueError("Invalid dataset_type. Choose either 'train' or 'val'.")

        # 업데이트된 self.graph_data.graphs를 사용하여 PyTorch Geometric 데이터 생성
        self.data_list = self.graph_data.convert_graphs_to_pyg()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# train_dataset = CustomDataset(root=data_path, dataset_type='train')
# val_dataset = CustomDataset(root=data_path, dataset_type='val')

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

