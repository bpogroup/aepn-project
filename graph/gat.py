import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6, concat=True):
        super(GATLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)

        alpha = softmax(alpha, edge_index_i, size_i)
        alpha = self.dropout(alpha)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        return aggr_out


class GATNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=8, heads=8, dropout=0.6):
        super(GATNet, self).__init__()

        self.conv1 = GATLayer(num_features, hidden_channels, heads, dropout)
        self.conv2 = GATLayer(heads * hidden_channels, num_classes, 1, dropout, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def softmax(src, index, num_nodes=None):
    out = src - src.max(dim=1, keepdim=True)[0]
    out = out.exp()
    out = scatter_add(out, index[0], dim=0, dim_size=num_nodes)
    return out / out.sum(dim=1, keepdim=True)

def scatter_add(src, index, dim=0, out=None, dim_size=None, fill_value=0):
    """Modified version of PyTorch Geometric's scatter_add to handle None dim_size."""
    #index_size = index.size(0) if index.dim() == 1 else index.size(dim)

    if dim_size is None:
        dim_size = index.max().item() + 1

    size = list(src.size())
    size[dim] = dim_size

    if out is None:
        out = src.new_full(size, fill_value)

    return out.scatter_add_(dim, index.view(-1, 1).expand_as(index), src)

# Example usage:
from torch_geometric.data import Data

# Generate a small example graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3 nodes, 16 features

# Create a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index)

# Create the GAT model
model = GATNet(num_features=16, num_classes=2, hidden_channels=8, heads=3, dropout=0.6)

# Forward pass
output = model(data)
print(output)