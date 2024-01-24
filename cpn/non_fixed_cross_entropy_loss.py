from typing import Optional

import numpy as np
from torch import Tensor
from torch_geometric.utils import softmax
import torch.nn as nn
import torch

from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


# Cross Entropy function.
def cross_entropy(y_pred, y_true, index, num_nodes: Optional[int] = None, dim: int = 0):
    # computing softmax values for predicted values
    y_pred = softmax(y_pred, index)

    temp = - np.log(y_pred) * y_true

    N = maybe_num_nodes(index, num_nodes)
    vec_loss = scatter(temp.detach(), index, dim, dim_size=N, reduce='sum')

    #loss = torch.mean(vec_loss)

    return vec_loss

if __name__ == '__main__':
    # y_true: True Probability Distribution
    y_true = torch.tensor([[0.5], [0], [0.5], [0], [1], [0], [0], [1]])
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

    # y_pred: Predicted values for each calss
    y_pred_logit = torch.tensor([[1], [0], [1], [0], [1], [0], [0], [1]])
    # y_pred = softmax(y_pred_logit, index)

    # Calling the cross_entropy function by passing
    # the suitable values
    cross_entropy_loss = cross_entropy(y_pred_logit, y_true, index)

    print("Cross Entropy Loss: ", cross_entropy_loss)