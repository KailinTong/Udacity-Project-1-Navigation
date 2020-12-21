import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") TODO
device = torch.device("cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)  # use GPU or not
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()  # evaluation
        with torch.no_grad():  # disabled gradient calculation
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # training

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        "*** YOUR CODE HERE ***"
        criterion = torch.nn.MSELoss()
        # Q target is a clone of Q network and generates targets y
        self.qnetwork_target.eval()
        # Q local is the Q network doing gradient descent
        self.qnetwork_local.train()

        # target q from target network
        target_q = rewards + gamma * torch.max(self.qnetwork_target(next_states), dim=1, keepdim=True)[0].detach().mul(
            1 - dones)

        expected_q = self.qnetwork_local(states).gather(1, actions)
        loss = criterion(expected_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DoubleQAgent(Agent):
    """Double Q network agent."""

    def __init__(self, state_size, action_size, seed, prioritized=False):
        super().__init__(state_size, action_size, seed)
        self.prioritized = prioritized

        if prioritized:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):

        # Save experience and priority value in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if self.prioritized is True:
                    idx_batch, experiences, ISWeights = self.memory.sample()
                    self.prioritized_learn(experiences, ISWeights, idx_batch, GAMMA)

                else:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    # noinspection PyMethodOverriding
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        update the memory and update weights with importance sampling weights

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        "*** YOUR CODE HERE ***"
        criterion = torch.nn.MSELoss()

        # Q target is a clone of Q network and generates targets y
        self.qnetwork_target.eval()
        # Q local is the Q network doing gradient descent
        self.qnetwork_local.eval()

        # action choose based on online network
        best_actions = torch.max(self.qnetwork_local(next_states), dim=1, keepdim=True)[1]
        # change back to train again
        self.qnetwork_local.train()

        # action evaluation based on target network to get q targets
        target_q = rewards + gamma * self.qnetwork_target(next_states).gather(1, best_actions).detach().mul(1 - dones)

        expected_q = self.qnetwork_local(states).gather(1, actions)
        loss = criterion(expected_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def prioritized_learn(self, experiences, ISWeights, idx_batch, gamma):
        """Update value parameters using given batch of experience tuples.
        update the prioritized memory and update weights with importance sampling weights

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        "*** YOUR CODE HERE ***"
        criterion = torch.nn.MSELoss()

        # Q target is a clone of Q network and generates targets y
        self.qnetwork_target.eval()
        # Q local is the Q network doing gradient descent
        self.qnetwork_local.train()

        # action choose
        best_actions = torch.max(self.qnetwork_target(next_states), dim=1, keepdim=True)[1]

        # action evaluation to get q targets
        target_q = rewards + gamma * self.qnetwork_target(next_states).gather(1, best_actions).detach().mul(1 - dones)

        expected_q = self.qnetwork_local(states).gather(1, actions)
        loss = criterion(expected_q * ISWeights, target_q * ISWeights)
        # delta = (expected_q - target_q ) * ISWeights
        # loss = torch.mean(torch.square(delta))

        abs_errors = torch.abs(expected_q - target_q)
        self.memory.batch_update(idx_batch, abs_errors.cpu().detach().numpy().squeeze())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


class DuelingQAgent(Agent):
    def __init__(self, state_size, action_size, seed, prioritized=False):
        """Dueling Q network agent."""
        super().__init__(state_size, action_size, seed)

        # Dueling Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)  # use GPU or not
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """ Prioritized buffer to store experience tuples."""

    abs_err_upper = 1.  # clipped abs error

    def __init__(self, action_size, buffer_size, batch_size, seed, epsilon=0.01, alpha=0.00, beta=0.0,
                 beta_increment_per_sampling=0.000):
        """Initialize a ReplayBuffer object.

        Args:
            action_size: dimension of each action
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            seed:  random seed
            epsilon: small amount to avoid zero priority
            alpha:  [0~1] convert the importance of TD error to priority
            beta: importance-sampling, from initial value increasing to 1
            beta_increment_per_sampling: beta increment
        """
        self.action_size = action_size
        self.memory = SumTree(capacity=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to prioritized memory."""
        max_p = np.max(self.memory.tree[-self.memory.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper

        e = self.experience(state, action, reward, next_state, done)
        self.memory.add(max_p, e)  # set the max p for new p

    def sample(self):
        """Sample a mini batch from prioritized memory."""
        idx_batch = np.empty((self.batch_size,), dtype=np.int32)
        ISWeights = np.empty((self.batch_size, 1), dtype=np.float32)

        pri_seg = self.memory.total_p / self.batch_size
        self.beta += self.beta_increment_per_sampling

        # for later calculate ISweight
        if self.memory.overflow is False:
            min_prob = np.min(self.memory.tree[
                              -self.memory.capacity:-self.memory.capacity + self.memory.data_pointer]) / self.memory.total_p
        else:
            min_prob = np.min(self.memory.tree[-self.memory.capacity:]) / self.memory.total_p

        s, a, r, ns, d = list(), list(), list(), list(), list()
        for i in range(self.batch_size):
            low_bd = pri_seg * i
            high_bd = pri_seg * (i + 1)
            v = np.random.uniform(low_bd, high_bd)
            idx, p, data = self.memory.get_leaf(v)
            prob = p / self.memory.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            idx_batch[i] = idx

            s.append(data.state)
            a.append(data.action)
            r.append(data.reward)
            ns.append(data.next_state)
            d.append(data.done)

        states = torch.from_numpy(np.vstack(s)).float().to(device)
        actions = torch.from_numpy(np.vstack(a)).long().to(device)
        rewards = torch.from_numpy(np.vstack(r)).float().to(device)
        next_states = torch.from_numpy(np.vstack(ns)).float().to(device)
        dones = torch.from_numpy(np.vstack(d).astype(np.uint8)).float().to(device)

        return idx_batch, (states, actions, rewards, next_states, dones), torch.from_numpy(ISWeights)

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.memory.update(ti, p)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.memory.capacity


class SumTree:
    """
    Store data with its priority in the tree.

    Reference code:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """

    data_pointer = 0
    overflow = False

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # dtype=object means that the array fills with pointers to python objects
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        """

        Args:
            p: priority value
            data: state transition tuple
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            self.overflow = True
            print("Sum tree memory overflow!")

    def update(self, tree_idx, p):
        """
        update the node priority through the tree
        Args:
            tree_idx: tree index
            p: priority
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]

        Args:
            v: priority value

        Returns:
            leaf_idx, self.tree[leaf_idx], self.data[data_idx]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root
