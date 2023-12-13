import networkx as nx
import numpy as np
from tianshou.data import Collector, ReplayBuffer, Batch
from gymnasium import spaces

b = ReplayBuffer(size=3)
#obs_1 = spaces.GraphInstance(nodes=np.array([[1], [2], [3]]), edges=None, edge_links=np.array([[0, 1], [1, 2]]))
#obs_2 = spaces.GraphInstance(nodes=np.array([[1], [2], [3]]), edges=None, edge_links=np.array([[0, 1], [1, 2]]))

obs = {"graph": spaces.GraphInstance(
                    nodes=np.array([[1], [2], [3]]),
                    edges=np.array([[1], [0]]),
                    edge_links=np.array([[0, 1], [1, 2]])),
       "mask": np.array([0, 1])}

# Create an empty graph
G = nx.Graph()

# Add nodes
G.add_node(1)
G.add_node(2)
G.add_node(3)

# Add edges
G.add_edge(1, 2)
G.add_edge(2, 3)

b.add(Batch(obs=nx.Graph(), act=0, rew=0, terminated=0, truncated=0, info={}))
print(b)