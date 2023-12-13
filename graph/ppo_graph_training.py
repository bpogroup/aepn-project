from typing import Union, Optional, Tuple, Dict
import networkx as nx
import tianshou
import torch
from torch_geometric.nn import GCNConv
from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import ReplayBuffer, Collector
#from collector import Collector


from graph.graph_deletion_env import GraphDeletionEnv

from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
import torch.multiprocessing as mp
import os

# Define a simple GAT model using PyTorch Geometric
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, actor=True):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 16, heads=1)
        self.conv2 = GATConv(16, 1, heads=1, concat=False)
        self.actor = actor
        self.graph = None

    def forward(self, observations, state: Optional[torch.Tensor] = None,
                info: Optional[Dict[str, int]] = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        #x = observations.x
        #edge_index = observations.edge_index
        x, edge_index = observations["graph_nodes"].squeeze(0), observations["graph_edge_links"].squeeze(0)

        #self.graph = observations

        # Convert the NetworkX graph to a PyTorch Geometric Data object
        #edge_index = from_networkx(self.graph).edge_index

        # Convert node attributes to a tensor
        #x = torch.tensor([[data['reward']] for _, data in self.graph.nodes(data=True)],
        #                 dtype=torch.float)


        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        if self.actor:
            x = x.reshape(x.shape[1], x.shape[0])
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x, state

# Define a simple GNN model using PyTorch Geometric
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data["x"], data["edge_index"]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class Critic(torch.nn.Module):
    def __init__(self, in_channels):
        super(Critic, self).__init__()
        self.conv1 = GATConv(in_channels, 16, heads=2, dropout=0.6)
        self.conv2 = GATConv(16 * 2, 1, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.sum(dim=0, keepdim=True)  # Sum over nodes

if __name__ == '__main__':
    def preprocess_function(**kwargs):
        if "obs" in kwargs:
            # print(kwargs["obs"][0]["graph"].edge_links.T)
            obs_with_tensors = [
                {"graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
                 "graph_edge_links": torch.from_numpy(obs["graph"].edge_links).int()}
                for obs in kwargs["obs"]]
            kwargs["obs"] = obs_with_tensors
        if "obs_next" in kwargs:
            obs_with_tensors = [
                {"graph_nodes": torch.from_numpy(obs["graph"][0]).float(),
                 "graph_edge_links": torch.from_numpy(obs["graph"][2]).int()}
                for obs in kwargs["obs_next"]]
            kwargs["obs_next"] = obs_with_tensors
        return kwargs


    num_envs = 1

    original_graph = nx.Graph()
    original_graph.add_nodes_from([(0, {'reward': 0}), (1, {'reward': 1}), (2, {'reward': 0})])
    original_graph.add_edges_from([(0, 1), (1, 2)])

    # Create the Gym environment
    venv = DummyVectorEnv([lambda : GraphDeletionEnv(original_graph) for _ in range(num_envs)])

    # Create a vectorized environment for more efficient training
    #venv = DummyVectorEnv([lambda: TaskAssignmentEnvironment(num_tasks=10, num_resources=5) for _ in range(num_envs)])

    # Define your policy using the GAT model and Tianshou's PPO policy
    model = GAT(in_channels=1, out_channels=1)
    # Define critic
    critic = Critic(in_channels=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy = PPOPolicy(model, critic, optim=optimizer, dist_fn=torch.distributions.categorical.Categorical, discount_factor=0.99, max_grad_norm=1.0)

    # Create a collector to collect data from the environment
    collector = Collector(policy, venv, tianshou.data.VectorReplayBuffer(10000, num_envs))#, preprocess_fn=preprocess_function)

    # Create a collector to collect test data from the environment
    test_collector = Collector(policy, venv)#, preprocess_fn=preprocess_function)
    #test_collector = None

    # Initialize the best reward as negative infinity
    best_reward = -float('inf')
    # Initialize a list to store the rewards
    rewards = []
    # Train the policy
    for epoch in range(10):
        trainer = tianshou.trainer.OnpolicyTrainer(
            policy,
            collector,
            test_collector=test_collector,
            max_epoch=1,
            step_per_epoch=5000,
            step_per_collect=200,
            update_per_step=0.25,
            repeat_per_collect=1,
            episode_per_test=10,
            batch_size=64,
            verbose=True,
        )
        result = trainer.run()
        # If the average reward of this epoch is better than the best reward so far, save the model
        if result['rew']:
            avg_reward = sum(result['rew']) / len(result['rew'])
            rewards.append(avg_reward)
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(policy.state_dict(), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best_model.pth'))

    import matplotlib.pyplot as plt

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.show()

    # Load the saved state dict into the policy
    policy.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best_model.pth')))

    # Create a test environment
    test_env = TaskAssignmentEnvironment(num_tasks=10, num_resources=5)

    # Create a collector for the test environment
    test_collector = Collector(policy, test_env)

    # Collect data from the test environment
    result = test_collector.collect(n_episode=100)

    # Calculate the average reward
    avg_reward = sum(result['rew']) / len(result['rew'])

    print(f'Average reward: {avg_reward}')