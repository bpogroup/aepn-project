import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from typing import List
import numpy as np

class Task():
    def __init__(self, task_type, compatibility):
        self.task_type = task_type
        self.compatibility = compatibility

class Resource():
    def __init__(self, resource_type, compatibility):
        self.resource_type = resource_type
        self.compatibility = compatibility #compatibility is a dictionary with key resource_type and value assignment duration
class TaskAssignmentEnvironment(gym.Env):
    """
    A simple task assignment environment where the agent has to assign tasks to resources through edge selection.
    The environment is fully observable and deterministic.
    """
    def __init__(self, num_tasks, num_resources):
        super(TaskAssignmentEnvironment, self).__init__()

        self.num_tasks = num_tasks
        self.num_resources = num_resources

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([num_tasks, num_resources])
        self.observation_space = spaces.Dict({
            'graph': spaces.Graph(node_space = spaces.Box(low=0, high=1, shape=(3,)), edge_space=spaces.Discrete(num_tasks + num_resources)),
            'assignments': spaces.MultiBinary(num_tasks * num_resources)
        })

        # Create an initial graph with no assignments
        self.graph = nx.complete_bipartite_graph(num_tasks, num_resources)
        self.assignments = []
        self.state = {'graph': self.graph, 'assignments': self.assignments}

    def reset(self):
        # Reset the environment to the initial state
        self.graph = nx.complete_bipartite_graph(self.num_tasks, self.num_resources)
        self.assignments = []
        self.state = {'graph': self.graph, 'assignments': self.assignments}
        return self.state, {}

    def step(self, action):
        # Perform the selected action (edge assignment)
        #WRONG
        #task_index = action // self.graph.number_of_nodes()  # task index
        #resource_index = action % self.graph.number_of_nodes() - self.graph.number_of_nodes() // 2  # resource index

        task_index = action[0]
        resource_index = action[1] + self.num_tasks

        action = (task_index, resource_index)

        if self.graph.has_edge(task_index, resource_index) and not action in self.assignments:
            # Edge exists and not already assigned, perform assignment
            self.graph.remove_edges_from([(task_index, resource_index)])
            # If nodes connected to the edge are now isolated, remove them
            self.graph.remove_nodes_from(list(nx.isolates(self.graph)))


            # # Update assignments
            self.assignments.append(action)

            # Update state
            self.state = {'graph': self.graph, 'assignments': self.assignments}

            # Calculate reward (simple reward for assignment, you might want to design a more complex one)
            reward = 1.0

            # Check if the graph is fully assigned (no remaining edges)
            done = len(self.graph.edges) == 0
        else:
            # Invalid action (trying to assign to an already assigned task or non-existent edge)
            reward = -1.0  # Penalize invalid actions
            done = False

        return self.state, reward, done, {}

    def action_masks(self) -> List[bool]:
        """
        Returns the list of possible edge indices that can be assigned
        """
        return [self.graph.has_edge(*edge) for edge in self.graph.edges]

    def render(self):
        # Optionally, you can implement a rendering function to visualize the state
        pass

if __name__ == '__main__':
    # Example Usage:
    env = TaskAssignmentEnvironment(num_tasks=3, num_resources=3)
    obs = env.reset()

    for _ in range(6):
        #sample action with uniform probability if action is in action_masks()
        action = [np.random.choice(np.arange(env.action_space[0].n)), np.random.choice(np.arange(env.action_space[1].n))]#[env.action_masks()])
        #action = env.action_space.sample()  # Replace with your RL agent's action
        next_obs, reward, done, _ = env.step(action)

        print("Action:", action, "Reward:", reward, "Done:", done)
        print("Current State - Assignments:", env.state['assignments'])
        print("Remaining Edges:", env.graph.edges)