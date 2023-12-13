import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data

from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

class GraphPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphPolicy, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size)
        self.conv2 = GATConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class PPOAgent:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.policy = GraphPolicy(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, data):
        logits = self.policy(data)
        action_probs = F.softmax(logits, dim=1)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[:, action]

    def train(self, rollout):
        states = torch.cat(rollout.states)
        actions = torch.tensor(rollout.actions)
        rewards = torch.tensor(rollout.rewards)
        action_probs = torch.cat(rollout.action_probs)

        returns = self.calculate_returns(rewards)

        old_probs = action_probs.gather(1, actions.unsqueeze(1))

        advantages = returns - rewards.mean()

        for _ in range(3):  # PPO optimization step
            new_probs = self.policy(states).gather(1, actions.unsqueeze(1))
            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def calculate_returns(self, rewards):
        gamma = 0.99
        running_add = 0
        returns = []

        for r in reversed(rewards):
            running_add = running_add * gamma + r
            returns.append(running_add)

        return torch.tensor(returns[::-1])

# Environment (dummy example)
class DynamicGraphEnvironment:
    def __init__(self, initial_nodes, initial_edges):
        self.nodes = initial_nodes
        self.edges = initial_edges

    def take_action(self, action):
        # Modify the graph structure based on the action
        # (In a real environment, this would involve updating the state of the environment)
        new_nodes = action[0]
        new_edges = action[1]
        self.nodes = new_nodes
        self.edges = new_edges

        # Return the new observation as a PyTorch Geometric Data object
        return Data(x=torch.randn(new_nodes, 16), edge_index=self.edges)

# Rollout storage
class Rollout:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []

    def add(self, state, action, reward, action_prob):
        self.states.append(state.x)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_probs.append(action_prob)

# Training loop
def train_ppo(agent, environment, num_episodes):
    for episode in range(num_episodes):
        state = environment.take_action((5, torch.tensor([[0, 1], [1, 0]])))
        rollout = Rollout()

        for _ in range(10):  # Episode length (adjust as needed)
            action, action_prob = agent.select_action(state)
            next_state = environment.take_action((action + 1, torch.tensor([[0, 1], [1, 0]])))  # Dummy action
            reward = torch.randn(1)  # Dummy reward

            rollout.add(state, action, reward, action_prob)
            state = next_state

        agent.train(rollout)
        print(f"Episode {episode + 1}/{num_episodes} completed.")

# Example usage
initial_nodes = 5
initial_edges = torch.tensor([[0, 1], [1, 0]])
env = DynamicGraphEnvironment(initial_nodes, initial_edges)
ppo_agent = PPOAgent(input_size=16, hidden_size=8, output_size=2)

train_ppo(ppo_agent, env, num_episodes=50)