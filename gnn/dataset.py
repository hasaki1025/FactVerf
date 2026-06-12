import torch
from torch_geometric.data import Data, Dataset
from typing import Any, Dict

from gnn.preprocess_dataset import FEVERDataSet

class FEVERGraphDataset(Dataset):
    """
    A PyG Dataset wrapper for FEVERDataSet.
    Converts the dictionary outputs from FEVERDataSet into PyG Data objects.
    """
    def __init__(self, fever_dataset: FEVERDataSet):
        super().__init__()
        self.fever_dataset = fever_dataset

    def len(self) -> int:
        return len(self.fever_dataset.data)

    def get(self, idx: int) -> Data:
        item = self.fever_dataset[idx]
        
        # PyG expects edge_index to be shape [2, num_edges]
        # and x to be shape [num_nodes, node_features]
        # and edge_attr to be shape [num_edges, edge_features]
        # FEVERDataSet outputs:
        # 'encode_node': Tensor[num_nodes, hidden_size]
        # 'encode_edge': Tensor[num_edges, hidden_size]
        # 'edge_indices': Tensor[2, num_edges]
        # 'label': scalar Tensor

        x = item['encode_node']
        edge_attr = item['encode_edge']
        edge_index = item['edge_indices']
        y = item['label'].view(1) # PyG prefers labels to have size [1] for proper batching

        # Construct a PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        return data
