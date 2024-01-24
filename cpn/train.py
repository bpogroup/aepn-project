#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import random
import json
import shutil

import numpy as np
import os

import torch
from torch_geometric.utils import scatter
from torch_geometric.nn import GCNConv, GATConv, to_hetero, HANConv
from torch_geometric.utils import softmax as pyg_softmax
from torch.nn.functional import softmax

# import AEPN_Env from new_aepn_env.py
from gym_env import new_aepn_env
try:
    from .gym_env import new_aepn_env as aepn_env
except ImportError:
    from gym_env import new_aepn_env as aepn_env

from gym_env.new_aepn_env import AEPN_Env
from pg import PGAgent, PPOAgent

import torch.nn.functional as F
import torch.nn as nn
train = True

def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train a new model",
                                     fromfile_prefix_chars='@')

    env = parser.add_argument_group('environment', 'environment type')
    env.add_argument('--environment',
                     choices=['ActionEvolutionPetriNetEnv'],
                     default='ActionEvolutionPetriNetEnv',
                     help='training environment')
    env.add_argument('--env_seed',
                     type=lambda x: int(x) if x.lower() != 'none' else None,
                     default=None,
                     help='seed for the environment')

    alg = parser.add_argument_group('algorithm', 'algorithm parameters')
    alg.add_argument('--algorithm',
                     choices=['ppo-clip', 'ppo-penalty', 'pg'],
                     default='ppo-clip',
                     help='training algorithm')
    alg.add_argument('--gam',
                     type=float,
                     default=0.99,
                     help='discount rate')
    alg.add_argument('--lam',
                     type=float,
                     default=0.97,
                     help='generalized advantage parameter')
    alg.add_argument('--eps',
                     type=float,
                     default=0.2,
                     help='clip ratio for clipped PPO')
    alg.add_argument('--c',
                     type=float,
                     default=0.01,
                     help='KLD weight for penalty PPO')
    alg.add_argument('--ent_bonus',
                     type=float,
                     default=0.0,
                     help='bonus factor for sampled policy entropy')
    alg.add_argument('--agent_seed',
                     type=lambda x: int(x) if x.lower() != 'none' else None,
                     default=None,
                     help='seed for the agent')

    policy = parser.add_argument_group('policy model')
    policy.add_argument('--policy_model',
                        choices=['gnn'],
                        default='gnn',
                        help='policy network type')
    policy.add_argument('--policy_kwargs',
                        type=json.loads,
                        default={"hidden_layers": [128]},
                        help='arguments to policy model constructor, passed through json.loads')
    policy.add_argument('--policy_lr',
                        type=float,
                        default=1e-4,
                        help='policy model learning rate')
    policy.add_argument('--policy_updates',
                        type=int,
                        default=40,
                        help='policy model updates per epoch')
    policy.add_argument('--policy_kld_limit',
                        type=float,
                        default=0.05, #0.01
                        help='KL divergence limit used for early stopping')
    policy.add_argument('--policy_weights',
                        type=str,
                        default="",#"policy-500.h5",
                        help='filename for initial policy weights')
    policy.add_argument('--policy_network',
                        type=str,
                        default="policy-500.pth",#"",
                        help='filename for initial policy weights')
    policy.add_argument('--score',
                        type = lambda x: str(x).lower() == 'true',
                        default = True,
                        help = 'have multi objective training')
    policy.add_argument('--score_weight',
                        type = float,
                        default=1e-3,
                        help='weight gradients of l2 loss')

    value = parser.add_argument_group('value model')
    value.add_argument('--value_model',
                       choices=['none', 'gnn'],
                       default='gnn',
                       help='value network type')
    value.add_argument('--value_kwargs',
                       type=json.loads,
                       default={"hidden_layers": [128]},
                       help='arguments to value model constructor, passed through json.loads')
    value.add_argument('--value_lr',
                       type=float,
                       default=1e-3,
                       help='the value model learning rate')
    value.add_argument('--value_updates',
                       type=int,
                       default=40,
                       help='value model updates per epoch')
    value.add_argument('--value_weights',
                       type=str,
                       default="",
                       help='filename for initial value weights')

    train = parser.add_argument_group('training')
    train.add_argument('--episodes',
                       type=int,
                       default=10, #100
                       help='number of episodes per epoch')
    train.add_argument('--epochs',
                       type=int,
                       default=50, #2500
                       help='number of epochs')
    train.add_argument('--max_episode_length',
                       type=lambda x: int(x) if x.lower() != 'none' else None,
                       default=100, #500
                       help='max number of interactions per episode')
    train.add_argument('--batch_size',
                       type=lambda x: int(x) if x.lower() != 'none' else None,
                       default=64,
                       help='size of batches in training')
    train.add_argument('--sort_states',
                       type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help='whether to sort the states before batching')
    train.add_argument('--use_gpu',
                       type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help='whether to use a GPU if available')
    train.add_argument('--verbose',
                       type=int,
                       default=0,
                       help='how much information to print')

    save = parser.add_argument_group('saving')
    save.add_argument('--name',
                       type=str,
                       default='run',
                       help='name of training run')
    save.add_argument('--datetag',
                       type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help='whether to append current time to run name')
    save.add_argument('--logdir',
                       type=str,
                       default='data/train',
                       help='base directory for training runs')
    save.add_argument('--save_freq',
                       type=int,
                       default=10,
                       help='how often to save the models')

    return parser

class ActorCritic(torch.nn.Module):
    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))

# Define a simple GNN model using PyTorch Geometric
class Actor(ActorCritic):
    def __init__(self, in_channels, out_channels):
        super(Actor, self).__init__()
        # self.conv1 = GCNConv(in_channels, 16)
        #self.conv2 = GCNConv(16, out_channels)
        self.conv1 = GATConv(in_channels, 16, heads=2)  # 2 attention heads
        self.conv2 = GATConv(16*2, out_channels)  # we multiply by 2 due to the 2 attention heads


    def forward(self, data):
        #check if data contains key 'graph'
        if 'graph' in data.keys():
            x, edge_index = torch.from_numpy(data['graph'].nodes).type(torch.float32), torch.from_numpy(data['graph'].edge_links)#data["x"], data["edge_index"]
        else:
            x = data['x'].float() #torch.cat(data['x'], dim=1)
            edge_index = data['edge_index']#data.edge_index
            index = data['index']

        x.requires_grad_()
        #edge_index.requires_grad = True

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        if 'graph' in data.keys():
            x = softmax(x, dim=0)
        else:
            x = pyg_softmax(x, index)
        return x





class HeteroActor(ActorCritic):
    def __init__(self, output_size, metadata):
        super(HeteroActor, self).__init__()
        self.conv1 = HANConv(-1, 16, heads=1, metadata=metadata)  # 2 attention heads
        self.conv2 = HANConv(16, output_size, metadata=metadata)  # we multiply by 2 due to the 2 attention heads

    def forward(self, data):
        if 'graph' in data.keys():
            #x, metadata = data['graph'], data['graph'].metadata()
            x = data['graph']
        else:
            x = data#data['x']
            #metadata = data['x'].metadata
            index = data['transition']['batch']
            #index=data['transition'].batch

        x_dict = x.x_dict
        edge_index_dict = x.edge_index_dict

        #import pdb; pdb.set_trace()
        x_dict = self.conv1(x_dict, edge_index_dict)#, edge_index).relu()
        x_dict = {k: F.relu(v) for k, v in x_dict.items() if v is not None}
        x_dict = self.conv2(x_dict, edge_index_dict)#, edge_index)
        if 'graph' in data.keys():
            if 'mask' in data.keys():
                mask = data['mask']['transition'].x
                mask = mask.masked_fill(mask == 0, float('-inf'))
                mask = mask.masked_fill(mask == 1, 0)
                #mask brings the values to -inf if the action is not valid

                #import pdb; pdb.set_trace()
                #x_dict's transition node is brought to -inf if the action is not valid
                x_dict['transition'] = x_dict['transition'] + mask

            #x_dict = {k: softmax(v, dim=0) for k, v in x_dict.items() if v is not None}
            x_dict = softmax(x_dict['transition'], dim=0)
        else:
            #x_dict = {k: pyg_softmax(v, index) for k, v in x_dict.items() if v is not None}
            #import pdb; pdb.set_trace()
            x_dict = pyg_softmax(x_dict['transition'], index)
        return x_dict

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        # Initialize parameters with the correct dimensions (TODO: make it authomatic)
        self.conv1.proj.arrival.weight = torch.nn.Parameter(torch.empty(16, 2))
        self.conv1.proj.waiting.weight = torch.nn.Parameter(torch.empty(16, 2))
        self.conv1.proj.resources.weight = torch.nn.Parameter(torch.empty(16, 4))
        self.conv1.proj.busy.weight = torch.nn.Parameter(torch.empty(16, 5))
        self.conv1.proj.transition.weight = torch.nn.Parameter(torch.empty(16, 1))


class HeteroCritic(ActorCritic):
    def __init__(self, output_size, metadata):
        super(HeteroCritic, self).__init__()
        self.conv1 = HANConv(-1, 16, heads=1, metadata=metadata)  # 2 attention heads
        self.conv2 = HANConv(16, output_size, metadata=metadata)  # we multiply by 2 due to the 2 attention heads
        self.lin = nn.Linear(16, 1)

    def forward(self, data):
        #x, edge_index = data['graph'].nodes.type(torch.float32), data['graph'].edge_links
        if 'graph' in data.keys():
            x, metadata = data['graph'], data['graph'].metadata()
        else:
            x = data
            index = data['transition']['batch']


        x_dict = x.x_dict
        edge_index_dict = x.edge_index_dict

        x_dict = self.conv1(x_dict, edge_index_dict)#, edge_index).relu()
        x_dict = {k: F.relu(v) for k, v in x_dict.items() if v is not None}
        if 'graph' in data.keys():
            #TODO: action masking
            #import pdb; pdb.set_trace()
            x_dict = {k: v.sum(dim=0, keepdim=True) for k, v in x_dict.items() if v is not None}
            x = sum(x_dict.values()) #simple aggregation

        else:
            #x_dict = {k: scatter(v, index, dim=0, reduce='sum') for k, v in x_dict.items() if v is not None}
            x_dict = scatter(x_dict['transition'], index, dim=0, reduce='sum')
            x = x_dict

        x = self.lin(x)
        #out = torch.mean(torch.stack([v for v in x_dict.values()]), dim=0) #TODO: check if this is the correct way to aggregate the outputs of the different node types
        return x

        #x = self.conv1(x.x_dict, x.edge_index_dict)
        #x = self.conv2(x.x_dict, x.edge_index_dict)
        #x = x.sum(dim=0, keepdim=True) #TODO: reintroduce index in the observation

        #return x_dict

class Critic(ActorCritic):
    def __init__(self, in_channels):
        super(Critic, self).__init__()
        self.conv1 = GATConv(in_channels, 16, heads=2, dropout=0.6)
        self.conv2 = GATConv(16 * 2, 1, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        if 'graph' in data.keys():
            x, edge_index = torch.from_numpy(data['graph'].nodes).type(torch.float32), torch.from_numpy(
                data['graph'].edge_links)
        else:
            x = data['x'].float() #torch.cat(data['x'], dim=1)
            edge_index = data['edge_index']#data.edge_index
            index = data['index']

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        if 'graph' in data.keys():
            x = x.sum(dim=0, keepdim=True)#.detach()  # Sum over nodes
        else:
            x = scatter(x, index, dim=0, reduce='sum')#.detach()
        return x  # Extract single output from tensor




def make_env(args, aepn = None):
    """Return the training environment for this run."""
    if args.environment == 'ActionEvolutionPetriNetEnv':
        env = AEPN_Env(aepn)
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    env.seed(args.env_seed)
    return env


def make_policy_network(args, metadata=None):
    """Return the policy network for this run."""
    if args.environment == 'ActionEvolutionPetriNetEnv':
        policy_network = HeteroActor(1, metadata=metadata)
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    if args.policy_weights != "":
        policy_network.load_weights(os.path.join(args.logdir, args.name, args.policy_weights))
    return policy_network


def make_value_network(args, metadata=None):
    """Return the value network for this run."""
    if args.value_model == 'none':
        value_network = None
    elif args.environment == 'ActionEvolutionPetriNetEnv':
        value_network = HeteroCritic(1, metadata=metadata)
    else:
        raise Exception("Unknown environment! Are you sure it is spelled correctly?")
    if args.value_weights != "":
        value_network.load_weights(args.value_weights)
    return value_network


def make_agent(args, metadata=None):
    """Return the agent for this run."""
    policy_network = make_policy_network(args, metadata=metadata)
    value_network = make_value_network(args, metadata=metadata)
    #currently using HANConv instead of to_hetero
    #if node_types and edge_types:
    #    policy_network = to_hetero(policy_network, node_types, edge_types)
    #    value_network = to_hetero(value_network, node_types, edge_types)

    if args.algorithm == 'pg':
        agent = PGAgent(policy_network=policy_network,policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                        value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                        gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit, ent_bonus=args.ent_bonus)
    elif args.algorithm == 'ppo-clip':
        agent = PPOAgent(policy_network=policy_network, method='clip', eps=args.eps,
                         policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                         value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                         gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit, ent_bonus=args.ent_bonus)
    elif args.algorithm == 'ppo-penalty':
        agent = PPOAgent(policy_network=policy_network, method='penalty', c=args.c,
                         policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                         value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                         gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit, ent_bonus=args.ent_bonus)
    return agent


def make_logdir(args):
    """Return the directory name for this run."""
    run_name = args.name
    if args.datetag:
        time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = time_string + '_' + run_name
    logdir = os.path.join(args.logdir, run_name)
    #make dir if it does not exist already
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write('--' + arg + '\n')
            if isinstance(value, dict):
                f.write(json.dumps(value) + "\n")
            else:
                f.write(str(value) + '\n')
    return logdir


if __name__ == '__main__':
    args = make_parser().parse_args()
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.agent_seed is not None:
        np.random.seed(args.agent_seed)
        random.seed(args.agent_seed)
        torch.manual_seed(args.agent_seed)
        torch.cuda.manual_seed(args.agent_seed)
        #TODO: two more lines for cuda

    env = make_env(args)
    agent = make_agent(args)
    # Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir .
    # then, in a browser page, access localhost:6006 to see the board
    logdir = make_logdir(args)
    print("Saving run in", logdir)
    if train:
        agent.train(env, episodes=args.episodes, epochs=args.epochs,
                    save_freq=args.save_freq, logdir=logdir, verbose=args.verbose,
                    max_episode_length=args.max_episode_length, batch_size=args.batch_size)
    else:
        #shutil.copy(args.policy_weights, os.path.join(logdir, "policy.h5"))
        with open(os.path.join(logdir, "results.csv"), 'w') as f:
            f.write("Return,Length\n")
        for _ in range(args.episodes):
            reward, length = agent.run_episode(env, max_episode_length=args.max_episode_length)
            with open(os.path.join(logdir, "results.csv"), 'a') as f:
                f.write(f"{reward},{length}\n")

