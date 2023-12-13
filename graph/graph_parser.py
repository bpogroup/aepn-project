import os

from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Union

import torch

import sys

from cpn.cpn_ast import PetriNetCreatorVisitor
from cpn.pn_parser import PNParser

sys.path.append("..")
from cpn.petrinet import AEPetriNet
class AEPNGraphParser:
    def __init__(self) -> None:
        #initialize the parser
        pass

    def net_to_graph(self, pn: AEPetriNet, max_size = 1024) -> HeteroData:
        """
        This function ingests an (initialized) A-E PN and outputs the corresponding graph, where tokens and transitions are embedded as nodes.
        Tokens in places that have arcs connected to a given transition will have (unlabeled) edges directed as the the original arc.
        This method assumes a maximum number of nodes (e.g. transitions + tokens) in the graph.
        """
        data = HeteroData()

        # Nodes represent places and transitions
        nodes = pn.places + pn.transitions

        # Create node feature matrix
        x = torch.eye(len(nodes))

        # Define edge indices based on arcs
        edge_indices = []
        for arc in pn.arcs:
            place, transition = arc
            place_index = nodes.index(place)
            transition_index = nodes.index(transition)
            edge_indices.append((place_index, transition_index))

        # Create the edge index tensor
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Create PyTorch Geometric Data object
        data = HeteroData(x=x, edge_index=edge_index)

        return data

    def graph_to_net(self, graph: HeteroData) -> AEPetriNet:
        """
        This function ingests a graph and returns the corresponding Action-Evolution Petri Net.
        Since the conversion procedure is costly, the original A-E PN is saved as a parameter of the parser and modified on the fly every time a 
        """
        pass

if __name__ == '__main__':
    # Create some node features
    node_features_2d = torch.randn(10, 2)  # 10 nodes with 2 features each
    node_features_3d = torch.randn(15, 3)  # 15 nodes with 3 features each

    # Create some edge indices
    edge_index_2d = torch.randint(0, 10, (2, 20))  # 20 edges between the 2D nodes
    edge_index_3d = torch.randint(0, 15, (2, 30))  # 30 edges between the 3D nodes

    edge_index_2d_3d = torch.randint(0, 25, (2, 40))  # 40 edges between the 2D and 3D nodes

    # Create the HeteroData object
    data = HeteroData()

    # Add the 2D nodes and edges
    data['2D_node'].x = node_features_2d
    data['2D_node_2D_node'].edge_index = edge_index_2d

    # Add the 3D nodes and edges
    data['3D_node'].x = node_features_3d
    data['3D_node_3D_node'].edge_index = edge_index_3d

    # Add the edges between 2D and 3D nodes
    data['2D_node_3D_node'].edge_index = edge_index_2d_3d

    # Create a pn
    f = open(os.path.join('..', 'networks', 'task_assignment_soft_comp.txt'), 'r')
    mynet_txt = f.read()
    f.close()
    pn_ast = PNParser().parse(mynet_txt)

    # parse additional_functions
    f = open(os.path.join('..', 'cpn', 'gym_env', 'additional_functions', 'color_functions.py'),
             'r')  # read functions file as txt
    my_functions = f.read()
    f.close()

    # create pn
    pn = PetriNetCreatorVisitor(net_type="AEPetriNet", additional_functions=my_functions).create(pn_ast)
    print(pn)


    parser = AEPNGraphParser()
