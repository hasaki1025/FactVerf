import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import yaml

class KnowledgeAugmentedGraphAttentionLayer(MessagePassing):
    def __init__(self, hidden_dim: int, heads: int = 8, leaky_relu_slope: float = 0.2):
        # We use 'add' aggregation as per the paper: sum over neighbors
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        
        # \Theta projections for node and edge features
        self.theta = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Attention mechanism vector a^T (one for each head)
        self.a = nn.Parameter(torch.Tensor(1, heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x, edge_index, edge_attr):
        # Add self loops to edge_index and edge_attr (fill with 0s)
        # This allows computing \gamma_{i,i} naturally
        num_nodes = x.size(0)
        edge_index, edge_attr = add_self_loops(
            edge_index, 
            edge_attr=edge_attr, 
            fill_value=0.0, 
            num_nodes=num_nodes
        )

        # Apply \Theta transformation
        x_proj = self.theta(x)
        edge_attr_proj = self.theta_edge(edge_attr)

        # Propagate messages
        out = self.propagate(edge_index, x=x_proj, edge_attr=edge_attr_proj)
        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # x_i: target node features [num_edges, hidden_dim]
        # x_j: source node features [num_edges, hidden_dim]
        # edge_attr: edge features [num_edges, hidden_dim]
        
        # Reshape for multi-head attention: [num_edges, heads, head_dim]
        x_i = x_i.view(-1, self.heads, self.head_dim)
        x_j = x_j.view(-1, self.heads, self.head_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.head_dim)
        
        # Concatenate [v_i || v_j || r_{ij}]
        concat_features = torch.cat([x_i, x_j, edge_attr], dim=-1) # [num_edges, heads, 3 * head_dim]
        
        # Compute attention scores e_{ij} = a^T LeakyReLU( [v_i || v_j || r_{ij}] )
        e_ij = (concat_features * self.a).sum(dim=-1) # [num_edges, heads]
        e_ij = self.leaky_relu(e_ij)
        
        # Softmax over neighborhood
        alpha = softmax(e_ij, index, ptr, size_i) # [num_edges, heads]
        
        # \gamma_{i,j} \Theta v_j
        return (x_j * alpha.unsqueeze(-1)).view(-1, self.hidden_dim)


class LLMSKAN(nn.Module):
    def __init__(self, config_path: str, config_override: dict = None):
        """
        Args:
            config_path: Path to the YAML configuration file.
            config_override: Optional dict of model hyperparameters that, when
                provided, take priority over values in the YAML file.  This is
                used by the Optuna tuning script (tune.py) to inject trial
                hyperparameters without writing a temporary YAML file.
                Supported keys: hidden_dim, num_layers, leaky_relu_slope.
        """
        super().__init__()
        
        # Load hyperparameters from yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)['model']

        # Apply optional overrides (e.g., from an Optuna trial)
        if config_override:
            config.update(config_override)
            
        input_dim = config.get('input_dim', 768)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 2)
        num_classes = config.get('num_classes', 3)
        leaky_relu_slope = config.get('leaky_relu_slope', 0.2)
        heads = config.get('heads', 8)
        
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            KnowledgeAugmentedGraphAttentionLayer(hidden_dim, heads, leaky_relu_slope)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        """
        Forward pass for PyG Batch.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_proj(x)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_proj(edge_attr)
        else:
            edge_attr = torch.zeros((0, self.node_proj.out_features), device=x.device)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Extract the claim node representation for each graph in the batch.
        # As implemented in preprocess_dataset, the claim node is always the first node (index 0).
        # PyG DataLoader provides `ptr`, which holds the start index of each graph in the batch.
        if hasattr(data, 'ptr') and data.ptr is not None:
            claim_indices = data.ptr[:-1]
        else:
            # Fallback to computing ptr if it is not explicitly available
            _, counts = torch.unique_consecutive(batch, return_counts=True)
            ptr = torch.cat([torch.tensor([0], device=x.device), counts.cumsum(0)])
            claim_indices = ptr[:-1]

        claim_repr = x[claim_indices]
        
        # Make predictions
        out = self.classifier(claim_repr)
        return out