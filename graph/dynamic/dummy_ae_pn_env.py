from typing import Any
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import numpy as np


def setup_example_graph_env():
    original_graph = nx.Graph()

    original_graph.add_nodes_from([(0, {'reward': 0}), (1, {'reward': 1}), (2, {'reward': 0})])
    original_graph.add_edges_from([(0, 1), (1, 2)])

    graph = original_graph

    return graph  # len(original_graph.nodes), len(original_graph.edges)

class DummyAEPNEnv(gym.Env):
    def __init__(self, graph=setup_example_graph_env()):
        super(DummyAEPNEnv, self).__init__()

        # Define the action space as the nodes in the graph
        self.action_space = spaces.Discrete(len(graph.nodes))

        # Define the observation space
        self.observation_space = spaces.Dict(
            {
                'graph': spaces.Graph(node_space=spaces.Box(low=0, high=1, shape=(1,)), edge_space=None)
            }
        )

        # Set the initial state
        self.graph = graph
        self.original_graph = graph.copy()

    def set_graph(self, graph):
        self.graph = graph

    def preprocess_function(self, obs):
        obs_with_tensors = {"graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
                            "graph_edge_links": torch.from_numpy(obs["graph"].edge_links).int()}
        return obs_with_tensors

    def _get_obs(self):
        # Return the current state of the graph as a space.Graph object
        # return Data(x=torch.tensor([self.graph.nodes[n]['reward'] for n in self.graph.nodes], dtype=torch.float), edge_index=torch.tensor(self.graph.edges, dtype=torch.long).transpose(0, 1))

        # Convert the NetworkX graph to a gym GraphInstance object
        pyg_g = from_networkx(self.graph)

        edge_index = np.array(pyg_g.edge_index)

        # Convert node attributes to a tensor
        x = np.array([[data['reward']] for _, data in self.graph.nodes(data=True)])

        # Create a Data object
        # data = Data(x=x, edge_index=edge_index)
        #obs = Batch.from_data_list([spaces.GraphInstance(nodes=x, edges=None, edge_links=edge_index)])

        #return obs
        return {'graph': spaces.GraphInstance(nodes=x, edges=None, edge_links=edge_index)}

    def step(self, action):
        # Check if the selected action (node) exists in the graph
        if action not in list(self.graph.nodes):
            raise ValueError(f"Invalid action: Node {action} does not exist in the graph.")
        if isinstance(action, np.ndarray):
            action = int(action)

        # Get the attribute value of the selected node
        reward = self.graph.nodes[action]['reward']

        # Remove the selected node and its direct neighbors from the graph
        nodes_to_remove = [action]  # + list(self.graph.neighbors(action))
        # nodes_to_remove = list(nx.single_source_shortest_path_length(self.graph, source=action).keys())

        self.graph.remove_nodes_from(nodes_to_remove)
        print(f"Removed nodes {nodes_to_remove}, generated reward {reward}")
        # Remove the edges between the selected node and its direct neighbors (not necessary)
        # self.graph.remove_edges_from([(action, neighbor) for neighbor in self.graph.neighbors(action)])

        # Update node indices
        self.graph = nx.convert_node_labels_to_integers(self.graph)

        # Check if the graph is empty
        done = len(self.graph.edges) == 0

        # If the graph is not empty, change the action space to a discrete space with the remaining nodes
        if not done:
            self.action_space = spaces.Discrete(len(self.graph.nodes))

        # Return the new state, reward, and whether the episode is done
        return self._get_obs(), reward, done, False, {}

    def reset(self, *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        # Reset the environment by restoring the original graph
        print('Resetting the environment')
        self.graph = self.original_graph.copy()
        self.action_space = spaces.Discrete(len(self.graph.nodes))
        return self._get_obs()

    def render(self, mode='human'):
        # Optional: render the current state of the graph
        pass

    def seed(self, env_seed):
        pass



if __name__ == '__main__':
    # Example usage:
    # Create a graph with attributes
    original_graph = nx.Graph()

    original_graph.add_nodes_from([(0, {'reward': 0}), (1, {'reward': 1}), (2, {'reward': 0})])
    original_graph.add_edges_from([(0, 1), (1, 2)])

    # Create the Gym environment
    env = DummyAEPNEnv(original_graph)
    done = False
    # Example episode loop
    while not done:
        action = env.action_space.sample()  # Replace with your RL agent's action
        observation, reward, done, truncated, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    # Reset the environment for a new episode
    env.reset()
