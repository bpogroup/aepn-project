"""
Original code: https://github.com/dylanpeifer/deepgroebner/tree/master
Rightfully referenced in the paper: https://arxiv.org/abs/2006.11287
"""

"""Policy gradient agents that support changing state spaces, specifically for graph environments.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy gradient) and proximal policy optimization
agent.
"""

import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
#from non_fixed_cross_entropy_loss import 

torch.autograd.set_detect_anomaly(True)

class GraphDataLoader(torch.utils.data.Dataset):
    def __init__(self, batch_size, states, actions, logprobs, advantages, values, data_type='hetero'):
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values

        if data_type == 'hetero':
            self.data_list = []
            for index in range(len(states)):
                temp_h_data = states[index]['graph']
                temp_h_data.y = actions[index]
                temp_h_data.reward = advantages[index]
                temp_h_data.logprobs = logprobs[index]
                temp_h_data.value = values[index]
                temp_h_data.mask = states[index]['mask']['a_transition'].x
                self.data_list.append(temp_h_data)
        elif data_type == 'homogeneous':
            self.data_list = [Data(x=torch.from_numpy(states[index]['graph'].nodes), edge_index=torch.from_numpy(states[index]['graph'].edge_links),
                             y=actions[index], reward=advantages[index], logprobs=logprobs[index], value=values[index])
                        for index in range(len(states))]
        else:
            raise ValueError("data_type must be either 'hetero' or 'homogeneous'")
        self.batch = Batch.from_data_list(self.data_list)
        self.loader = DataLoader(self.data_list, batch_size=batch_size)

    def __getitem__(self, index):
        # Convert the data to a PyTorch Geometric Data object
        data = Data(x=self.states[index][0], edge_index=self.states[index][1], y=self.actions[index],
                    reward=self.advantages[index], logprobs=self.logprobs[index], value=self.values[index])

        return data

    def __len__(self):
        return len(self.data_list)

def discount_rewards(rewards, gam):
    """Return discounted rewards-to-go computed from inputs.

    Parameters
    ----------
    rewards : array_like
        List or 1D array of rewards from a single complete trajectory.
    gam : float
        Discount rate.

    Returns
    -------
    rewards : ndarray
        1D array of discounted rewards-to-go.

    Examples
    --------
    >>> rewards = [1, 2, 3, 4, 5]
    >>> discount_rewards(rewards, 0.5)
    [1, 2, 6.25, 6.5, 5]

    """
    cumulative_reward = 0
    discounted_rewards = rewards.clone()
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gam * cumulative_reward
        discounted_rewards[i] = cumulative_reward
    return discounted_rewards


def compute_advantages(rewards, values, gam, lam):
    """Return generalized advantage estimates computed from inputs.

    Parameters
    ----------
    rewards : array_like
        List or 1D array of rewards from a single complete trajectory.
    values : array_like
        List or 1D array of value predictions from a single complete trajectory.
    gam : float
        Discount rate.
    lam : float
        Parameter for generalized advantage estimation.

    Returns
    -------
    advantages : ndarray
        1D array of computed advantage scores.

    References
    ----------
    .. [1] Schulman et al, "High-Dimensional Continuous Control Using
       Generalized Advantage Estimation," ICLR 2016.

    Examples
    --------
    >>> rewards = [1, 1, 1, 1, 1]
    >>> values = [0, 0, 0, 0, 0]
    >>> compute_advantages(rewards, values, 0.5, 0.5)
    array([1.33203125, 1.328125  , 1.3125    , 1.25      , 1.        ])

    """
    #rewards = np.array(rewards, dtype=np.float32)
    #values = values.detach().numpy()
    #values = np.array(values, dtype=np.float32)
    delta = rewards - values
    delta[:-1] += gam * delta[1:]#values[1:]
    return discount_rewards(delta, gam * lam)


class TrajectoryBuffer:
    """A buffer to store and compute with trajectories.

    The buffer is used to store information from each step of interaction
    between the agent and environment. When a trajectory is finished it
    computes the discounted rewards and generalized advantage estimates. After
    some number of trajectories are finished it can return a tf.Dataset of the
    training data for policy gradient algorithms.

    Parameters
    ----------
    gam : float, optional
        Discount rate.
    lam : float, optional
        Parameter for generalized advantage estimation.

    See Also
    --------
    discount_rewards : Discount the list or array of rewards by gamma in-place.
    compute_advantages : Return generalized advantage estimates computed from inputs.

    """

    def __init__(self, gam=0.99, lam=0.97, data_type = 'hetero', action_mode="node_selection"):
        self.gam = gam
        self.lam = lam
        self.states = []
        self.actions = []
        self.rewards = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.logprobs = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.values =  torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.start = 0  # index to start of current episode
        self.end = 0  # index to one past end of current episode
        self.action_mode = action_mode # "node_selection" or "edge_selection"

        self.data_type = data_type

        #logging utilities
        self.prev_policy_loss = 0


    def store(self, state, action, reward, logprob, value):
        """Store the information from one interaction with the environment.

        Parameters
        ----------
        state : ndarray
           Observation of the state.
        action : int
           Chosen action in this trajectory.
        reward : float
           Reward received in the next transition.
        logprob : float
           Agent's logged probability of picking the chosen action.
        value : float
           Agent's computed value of the state.

        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards = torch.cat((self.rewards, torch.tensor([reward])))
        self.logprobs = torch.cat((self.logprobs, torch.tensor([logprob])))
        self.values = torch.cat((self.values, torch.tensor([value])))
        self.end += 1
        #import pdb; pdb.set_trace()

    def finish(self):
        """Finish an episode and compute advantages and discounted rewards.

        Advantages are stored in place of `values` and discounted rewards are
        stored in place of `rewards` for the current trajectory.
        """
        #convert lists of tensors to tensors
        #self.values = torch.stack(self.values)
        #self.rewards = torch.tensor(self.rewards, dtype=torch.float32)


        tau = slice(self.start, self.end)
        rewards = discount_rewards(self.rewards[tau], self.gam)
        values = compute_advantages(self.rewards[tau], self.values[tau], self.gam, self.lam)
        self.rewards[tau] = rewards
        self.values[tau] = values
        self.start = self.end

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.logprobs = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.values = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.start = 0
        self.end = 0

    def get(self, batch_size=64, normalize_advantages=False, sort=False, drop_remainder=False):
        """Return a tf.Dataset of training data from this TrajectoryBuffer, along with the desired batch size.

        Parameters
        ----------
        batch_size : int, optional
            Batch size in the returned tf.Dataset.
        normalize_advantages : bool, optional
            Whether to normalize the returned advantages.
        sort : bool, optional
            Whether to sort by state shape before batching to minimize padding.
        drop_remainder : bool, optional
            Whether to drop the last batch if it has fewer than batch_size elements.

        Returns
        -------
        dataset : tf.Dataset
        batch_size : int

        """
        actions = np.array(self.actions[:self.start], dtype=np.int32)
        logprobs = self.logprobs[:self.start]
        advantages = self.values[:self.start]
        values = self.rewards[:self.start]

        if normalize_advantages:
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        if self.states: #and self.states[0].ndim == 2:

            # filter out any states with only one action available
            if self.action_mode == "node_selection":
                if self.data_type == 'hetero':
                    #no need to filter anything
                    states = self.states[:self.start]
                    pass

                elif self.data_type == 'homogeneous':
                    indices = [i for i in range(len(self.states[:self.start])) if len(self.states[i]['graph'].nodes) != 1]
                    states = [self.states[i] for i in indices]
                    actions = actions[indices]
                    logprobs = logprobs[indices]
                    advantages = advantages[indices]
                    values = values[indices]

                    if sort:
                        indices = np.argsort([s.shape[0] for s in states])
                        states = [states[i] for i in indices]
                        actions = actions[indices]
                        logprobs = logprobs[indices]
                        advantages = advantages[indices]
                        values = values[indices]
            elif self.action_mode == "edge_selection":
                indices = [i for i in range(len(self.states[:self.start])) if len(self.states[0]['graph'].edge_links) != 1]
            else:
                raise ValueError("Action_mode must be either 'node_selection' or 'edge_selection'")

            dataloader = GraphDataLoader(batch_size, states, actions, logprobs, advantages, values).loader
            #if batch_size is None:
            #    batch_size = len(states)

        else:
            raise ValueError("States must be non-empty.")

        return dataloader

    def __len__(self):
        return len(self.states)


def print_status_bar(i, epochs, history, verbose=1):
    """Print a formatted status line."""
    metrics = "".join([" - {}: {:.4f}".format(m, history[m][i])
                       for m in ['mean_returns']])
    end = "\n" if verbose == 2 or i+1 == epochs else ""
    print("\rEpoch {}/{}".format(i+1, epochs) + metrics, end=end)


class Agent:
    """Abstract base class for policy gradient agents.

    All functionality for policy gradient is implemented in this
    class. Derived classes must define the property `policy_loss`
    which is used to train the policy.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    policy_lr : float, optional
        The learning rate for the policy model.
    policy_updates : int, optional
        The number of policy updates per epoch of training.
    value_network : network, None, or string, optional
        The network for the value model.
    value_lr : float, optional
        The learning rate for the value model.
    value_updates : int, optional
        The number of value updates per epoch of training.
    gam : float, optional
        The discount rate.
    lam : float, optional
        The parameter for generalized advantage estimation.
    normalize_advantages : bool, optional
        Whether to normalize advantages.
    kld_limit : float, optional
        The limit on KL divergence for early stopping policy updates.
    ent_bonus : float, optional
        Bonus factor for sampled policy entropy.

    """

    def __init__(self,
                 policy_network, policy_lr=1e-4, policy_updates=1,
                 value_network=None, value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 kld_limit=0.01, ent_bonus=0.0, data_type='hetero'):
        self.policy_model = policy_network
        self.policy_loss = NotImplementedError
        self.policy_optimizer = torch.optim.Adam(params=list(policy_network.parameters()),
                                                 lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = torch.nn.MSELoss()
        self.value_optimizer = torch.optim.Adam(params=list(value_network.parameters()), lr=value_lr)
        self.value_updates = value_updates

        self.lam = lam
        self.gam = gam
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam)
        self.normalize_advantages = normalize_advantages
        self.kld_limit = kld_limit
        self.ent_bonus = ent_bonus

        self.data_type = data_type #TODO: make this univoque

        self.previous_policy_loss = 0

    def act(self, state, return_logprob=False):
        """Return an action for the given state using the policy model.

        Parameters
        ----------
        state : np.array
            The state of the environment.
        return_logprob : bool, optional
            Whether to return the log probability of choosing the chosen action.

        """
        self.policy_model.eval()  # set model to evaluation mode
        pi = self.policy_model(state)
        logpi = pi.log() #.unsqueeze(0))
        #import pdb; pdb.set_trace()
        action = torch.multinomial(torch.exp(logpi.squeeze(1)), 1)[0]#[0, 0] #TODO: implement deterministic policy
        print(f"Sampled action: {action}")
        #import pdb; pdb.set_trace()
        if return_logprob:
            return action.item(), logpi[action].item()
        else:
            return action.item()

    def value(self, state):
        """Return the predicted value for the given state using the value model.

        Parameters
        ----------
        state : np.array
            The state of the environment.

        """
        return self.value_model(state)[0][0]

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, batch_size=64, sort_states=False):
        """Train the agent on env.

        Parameters
        ----------
        env : environment
            The environment to train on.
        episodes : int, optional
            The number of episodes to perform per epoch of training.
        epochs : int, optional
            The number of epochs to train.
        max_episode_length : int, optional
            The maximum number of steps of interaction in an episode.
        verbose : int, optional
            How much information to print to the user.
        save_freq : int, optional
            How often to save the model weights, measured in epochs.
        logdir : str, optional
            The directory to store Tensorboard logs and model weights.
        batch_size : int or None, optional
            The batch sizes for training (None indicates one large batch).
        sort_states : bool, optional
            Whether to sort the states to minimize padding.

        Returns
        -------
        history : dict
            Dictionary with statistics from training.

        """
        tb_writer = None if logdir is None else SummaryWriter(log_dir=logdir)
        history = {'mean_returns': np.zeros(epochs),
                   'min_returns': np.zeros(epochs),
                   'max_returns': np.zeros(epochs),
                   'std_returns': np.zeros(epochs),
                   'mean_ep_lens': np.zeros(epochs),
                   'min_ep_lens': np.zeros(epochs),
                   'max_ep_lens': np.zeros(epochs),
                   'std_ep_lens': np.zeros(epochs),
                   'policy_updates': np.zeros(epochs),
                   'delta_policy_loss': np.zeros(epochs),
                   'policy_ent': np.zeros(epochs),
                   'policy_kld': np.zeros(epochs)}

        for i in range(epochs):

            self.buffer.clear()
            return_history = self.run_episodes(env, episodes=episodes, max_episode_length=max_episode_length, store=True)
            dataloader = self.buffer.get(normalize_advantages=self.normalize_advantages, batch_size=batch_size, sort=sort_states)
            #batch = self.buffer.get(normalize_advantages=self.normalize_advantages, batch_size=batch_size,sort=sort_states)
            policy_history = self._fit_policy_model(dataloader, epochs=self.policy_updates)
            value_history = self._fit_value_model(dataloader, epochs=self.value_updates)

            history['mean_returns'][i] = np.mean(return_history['returns'])
            history['min_returns'][i] = np.min(return_history['returns'])
            history['max_returns'][i] = np.max(return_history['returns'])
            history['std_returns'][i] = np.std(return_history['returns'])
            history['mean_ep_lens'][i] = np.mean(return_history['lengths'])
            history['min_ep_lens'][i] = np.min(return_history['lengths'])
            history['max_ep_lens'][i] = np.max(return_history['lengths'])
            history['std_ep_lens'][i] = np.std(return_history['lengths'])
            history['policy_updates'][i] = len(policy_history['loss'])
            history['delta_policy_loss'][i] = policy_history['loss'][-1] - self.previous_policy_loss
            self.previous_policy_loss = policy_history['loss'][-1] #dunno exactly what changed here, but this seems to work
            history['policy_ent'][i] = policy_history['ent'][-1]
            history['policy_kld'][i] = policy_history['kld'][-1]

            if logdir is not None and (i+1) % save_freq == 0:
                self.save_policy_weights(logdir + "/policy-" + str(i+1) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i+1) + ".h5")
                self.save_policy_network(logdir + "/network-" + str(i+1) + ".pth")
            if tb_writer is not None:
                tb_writer.add_scalar('mean_returns', history['mean_returns'][i], global_step=i)
                tb_writer.add_scalar('min_returns', history['min_returns'][i], global_step=i)
                tb_writer.add_scalar('max_returns', history['max_returns'][i], global_step=i)
                tb_writer.add_scalar('std_returns', history['std_returns'][i], global_step=i)
                tb_writer.add_scalar('mean_ep_lens', history['mean_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('min_ep_lens', history['min_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('max_ep_lens', history['max_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('std_ep_lens', history['std_ep_lens'][i], global_step=i)
                #tb_writer.add_histogram('returns', return_history['returns'], global_step=i) #BROKEN
                #tb_writer.add_histogram('lengths', return_history['lengths'], global_step=i)
                tb_writer.add_scalar('policy_updates', history['policy_updates'][i], global_step=i)
                tb_writer.add_scalar('delta_policy_loss', history['delta_policy_loss'][i], global_step=i)
                tb_writer.add_scalar('policy_ent', history['policy_ent'][i], global_step=i)
                tb_writer.add_scalar('policy_kld', history['policy_kld'][i], global_step=i)
                tb_writer.flush()
            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

        return history

    def run_episode(self, env, max_episode_length=None, buffer=None):
        """Run an episode and return total reward and episode length.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        max_episode_length : int, optional
            The maximum number of interactions before the episode ends.
        buffer : TrajectoryBuffer object, optional
            If included, it will store the whole rollout in the given buffer.

        Returns
        -------
        (total_reward, episode_length) : (float, int)
            The total nondiscounted reward obtained in this episode and the
            episode length.

        """
        state = env.reset()
        done = False
        episode_length = 0
        total_reward = 0
        while not done:
            #if state.dtype == np.float64:
            #    state = state.astype(np.float32)
            #import pdb; pdb.set_trace()
            action, logprob = self.act(state, return_logprob=True)
            if self.value_model is None:
                value = 0
            elif isinstance(self.value_model, str):
                value = env.value(strategy=self.value_model, gamma=self.gam)
            else:
                value = self.value(state)
            next_state, reward, done, truncated, _ = env.step(action)
            if buffer is not None:
                buffer.store(state, action, reward, logprob, value)
            episode_length += 1
            total_reward += reward
            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state
        if buffer is not None:
            buffer.finish()
        return total_reward, episode_length

    def run_episodes(self, env, episodes=100, max_episode_length=None, store=False):
        """Run several episodes, store interaction in buffer, and return history.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        episodes : int, optional
            The number of episodes to perform.
        max_episode_length : int, optional
            The maximum number of steps before the episode is terminated.
        store : bool, optional
            Whether or not to store the rollout in self.buffer.

        Returns
        -------
        history : dict
            Dictionary which contains information from the runs.

        """
        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}
        for i in range(episodes):
            R, L = self.run_episode(env, max_episode_length=max_episode_length, buffer=self.buffer)
            history['returns'][i] = R
            history['lengths'][i] = L
        return history

    def _fit_policy_model(self, dataloader, epochs=1):
        """Fit policy model using data from dataset."""
        history = {'loss': [], 'kld': [], 'ent': []}
        for epoch in range(epochs):
            loss, kld, ent, batches = 0, 0, 0, 0
            for batch in dataloader:
                print('Batch: ', batches, ' of ', len(dataloader))
                batch_loss, batch_kld, batch_ent = self._fit_policy_model_step(batch)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1
            history['loss'].append(loss / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
            if self.kld_limit is not None and kld > self.kld_limit:
                print(f'Early stopping at epoch {epoch+1} due to KLD divergence.')
                break
        return {k: np.array(v) for k, v in history.items()}

    def _fit_policy_model_step(self, batch):
        """Fit policy model on one batch of data."""
        self.policy_model.train()  # set model to training mode
        self.policy_optimizer.zero_grad()  # zero out gradients

        # Save the initial weights
        #initial_weights = {name: param.clone() for name, param in self.policy_model.named_parameters()}


       #
        if self.data_type == 'homogeneous':
            indexes = batch.batch.data
            states = {'x': batch.x, 'edge_index': batch.edge_index, 'index': batch.batch.data}
            actions = torch.tensor(batch.y)
            logprobs = batch.logprobs.clone()
            advantages = batch.rewards.clone()

        elif self.data_type == 'hetero':

            indexes = batch['a_transition'].batch.data
            states = batch
            actions = torch.tensor(batch.y)
            logprobs = batch.logprobs.clone()
            advantages = batch.reward.clone()

        epsilon = 1e-10
        logpis = (self.policy_model(states) + epsilon).log()
        #logpis = self.policy_model(states).log()
        #import pdb;
        #pdb.set_trace()
        #new_logprobs contains, for each unique index in indexes, the value in the slice of logpis corresponding
        #to the current index in indexes with index action[index]
        new_logprobs = torch.stack([logpis[indexes == index][actions[index]] for index in indexes.unique()]).squeeze(1)
        #import pdb; pdb.set_trace()
        probs = torch.exp(new_logprobs)
        # Compute the loss and gradients
        ent = -torch.sum(probs * new_logprobs)  # Correct entropy calculation
        loss = torch.mean(self.policy_loss(new_logprobs, logprobs, advantages)) - self.ent_bonus * ent



        kld = torch.sum(probs * (logprobs - new_logprobs))  # Correct KL divergence calculation


        #ent = -torch.mean(new_logprobs)
        #loss = torch.mean(self.policy_loss(new_logprobs, logprobs, advantages)) - self.ent_bonus * ent
        #kld = torch.mean(logprobs - new_logprobs)
        loss.backward(retain_graph=True)  # compute gradients

        # Print out the gradients of the parameters
        for name, param in self.policy_model.named_parameters():
            if param.grad is not None:
                print(f"Gradient of {name}: {param.grad}")


        self.policy_optimizer.step()

        #updated_weights = {name: param.clone() for name, param in self.policy_model.named_parameters()}
        # Compare the initial and updated weights
        #for name, initial in initial_weights.items():
        #    updated = updated_weights[name]
        #    if torch.all(torch.eq(initial, updated)):
        #        print(f'Weights of {name} did not change.')
        #    else:
        #        print(f'Weights of {name} changed.')


        return loss.item(), kld.item(), ent.item()

    def load_policy_weights(self, filename):
        """Load weights from filename into the policy model."""
        self.policy_model.load_weights(filename)

    def save_policy_weights(self, filename):
        """Save the current weights in the policy model to filename."""
        self.policy_model.save_weights(filename)

    def _fit_value_model(self, dataloader, epochs=1):
        """Fit value model using data from dataset."""
        if self.value_model is None or isinstance(self.value_model, str):
            epochs = 0
        history = {'loss': []}
        for epoch in range(epochs):
            loss, batches = 0, 0
            for batch in dataloader:
                batch_loss = self._fit_value_model_step(batch)
                loss += batch_loss
                batches += 1
            history['loss'].append(loss / batches)
        return {k: np.array(v) for k, v in history.items()}

    def _fit_value_model_step(self, batch):
        """Fit value model on one batch of data."""
        self.value_model.train()

        if self.data_type == 'homogeneous':
            indexes = batch.batch.data
            states = {'x': batch.x, 'edge_index': batch.edge_index, 'index': batch.batch.data}
            values = batch.value.clone() #TODO: is this correct?
        elif self.data_type == 'hetero':
            indexes = batch['a_transition'].batch.data
            states = batch
            values = batch.value.clone()

        pred_values = self.value_model(states).squeeze()
        loss = torch.mean(self.value_loss.forward(input=pred_values, target=values))

        self.value_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.value_optimizer.step()


        return loss.item()


    def load_value_weights(self, filename):
        """Load weights from filename into the value model."""
        if self.value_model is not None and self.value_model != 'env':
            self.value_model.load_weights(filename)

    def save_value_weights(self, filename):
        """Save the current weights in the value model to filename."""
        if self.value_model is not None and not isinstance(self.value_model, str):
            self.value_model.save_weights(filename)

    def save_policy_network(self, filename):
        """Save the current weights in the value model to filename."""
        torch.save(self.policy_model, filename)

    def load_policy_network(self, filename):
        """Save the current weights in the value model to filename."""
        self.policy_model = torch.load(torch.load(filename))

def pg_surrogate_loss(new_logps, old_logps, advantages):
    """Return loss with gradient for policy gradient.

    Parameters
    ----------
    new_logps : Tensor (batch_dim,)
        The output of the current model for the chosen action.
    old_logps : Tensor (batch_dim,)
        The previous logged probability of the chosen action.
    advantages : Tensor (batch_dim,)
        The computed advantages.

    Returns
    -------
    loss : Tensor (batch_dim,)
        The loss for each interaction.

    """
    return -new_logps * advantages


class PGAgent(Agent):
    """A policy gradient agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.

    """

    def __init__(self, policy_network, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = pg_surrogate_loss


def ppo_surrogate_loss(method='clip', eps=0.2, c=0.01):
    """Return loss function with gradient for proximal policy optimization.

    Parameters
    ----------
    method : {'clip', 'penalty'}
        The specific loss for PPO.
    eps : float
        The clip ratio if using 'clip'.
    c : float
        The fixed KLD weight if using 'penalty'.

    """
    if method == 'clip':

        def loss(new_logps, old_logps, advantages):
            """Return loss with gradient for clipped PPO.

            Parameters
            ----------
            new_logps : Tensor (batch_dim,)
                The output of the current model for the chosen action.
            old_logps : Tensor (batch_dim,)
                The previous logged probability for the chosen action.
            advantages : Tensor (batch_dim,)
                The computed advantages.

            Returns
            -------
            loss : Tensor (batch_dim,)
                The loss for each interaction.
            """
            min_adv = torch.where(advantages > 0, (1 + eps) * advantages, (1 - eps) * advantages)
            return -torch.min(torch.exp(new_logps - old_logps) * advantages, min_adv)
        return loss
    elif method == 'penalty':
        def loss(new_logps, old_logps, advantages):
            """Return loss with gradient for penalty PPO.

            Parameters
            ----------
            new_logps : Tensor (batch_dim,)
                The output of the current model for the chosen action.
            old_logps : Tensor (batch_dim,)
                The previous logged probability for the chosen action.
            advantages : Tensor (batch_dim,)
                The computed advantages.

            Returns
            -------
            loss : Tensor (batch_dim,)
                The loss for each interaction.
            """
            return -(torch.exp(new_logps - old_logps) * advantages - c * (old_logps - new_logps))
        return loss
    else:
        raise ValueError('unknown PPO method')


class PPOAgent(Agent):
    """A proximal policy optimization agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    method : {'clip', 'penalty'}
        The specific loss for PPO.
    eps : float
        The clip ratio if using 'clip'.
    c : float
        The fixed KLD weight if using 'penalty'.

    """

    def __init__(self, policy_network, method='clip', eps=0.2, c=0.01, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = ppo_surrogate_loss(method=method, eps=eps, c=c)
