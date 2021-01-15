import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN, DuelingDQN, MaxminDQN, CategoricalDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory

from mushroom_rl_benchmark.builders import AgentBuilder
from mushroom_rl_benchmark.builders.network import DQNNetwork, DQNFeatureNetwork


class DQNBuilder(AgentBuilder):
    """
    AgentBuilder for Deep Q-Network (DQN).

    """
    def __init__(self, policy, approximator, approximator_params, alg_params, n_steps_per_fit=1):
        """
        Constructor.

        Args:
            policy (Policy): policy class;
            approximator (dict): Q-function approximator;
            approximator_params (dict): parameters of the Q-function approximator;
            alg_params (dict): parameters for the algorithm;
            n_steps_per_fit (int, 1): number of steps per fit.

        """
        self.policy = policy
        self.approximator = approximator
        self.approximator_params = approximator_params
        self.alg_params = alg_params

        super().__init__(n_steps_per_fit, compute_policy_entropy=False)

    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return DQN(mdp_info, self.policy, self.approximator, self.approximator_params, **self.alg_params)

    def compute_Q(self, agent, states):
        q_max = agent.approximator(states).max()

        return q_max

    def set_eval_mode(self, agent, eval):
        if eval:
            agent.policy.set_epsilon(self.epsilon_test)
        else:
            agent.policy.set_epsilon(self.epsilon)

    @classmethod
    def default(cls, lr=.0001, network=DQNNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_steps_per_fit=1, use_cuda=False):
        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            loss=F.smooth_l1_loss,
            use_cuda=use_cuda)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)


class DoubleDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return DoubleDQN(mdp_info, self.policy, self.approximator, self.approximator_params, **self.alg_params)


class PrioritizedDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n

        replay_memory = PrioritizedReplayMemory(self.alg_params['initial_replay_size'],
            self.alg_params['max_replay_size'], alpha=.6,
            beta=LinearParameter(.4, threshold_value=1, n=50000000 // 4)
        )
        self.alg_params['replay_memory'] = replay_memory
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return DQN(mdp_info, self.policy, self.approximator, self.approximator_params, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=DQNNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_steps_per_fit=1, use_cuda=False):

        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            loss=F.smooth_l1_loss,
            use_cuda=use_cuda)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)


class AveragedDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.alg_params['approximator_params'] = self.approximator_params
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return AveragedDQN(mdp_info, self.policy, self.approximator, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=DQNNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_steps_per_fit=1, n_approximators=10, use_cuda=False):
        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            loss=F.smooth_l1_loss,
            use_cuda=use_cuda)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            n_approximators=n_approximators,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)


class DuelingDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return DuelingDQN(mdp_info, self.policy, self.approximator_params, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=DQNFeatureNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, use_cuda=False):
        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            n_features=n_features,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            loss=F.smooth_l1_loss,
            use_cuda=use_cuda)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)


class CategoricalDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.approximator_params['network'] = DQNFeatureNetwork
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return CategoricalDQN(mdp_info, self.policy, self.approximator_params, **self.alg_params)

    @staticmethod
    def categorical_loss(input, target):
        input = input.clamp(1e-5)

        return -torch.sum(target * torch.log(input))

    @classmethod
    def default(cls, lr=.0001, network=DQNFeatureNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, v_min=-10, v_max=10,
                n_atoms=51, use_cuda=False):

        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            n_features=n_features,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            loss=CategoricalDQNBuilder.categorical_loss,
            use_cuda=use_cuda)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)


class MaxminDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.alg_params['approximator_params'] = self.approximator_params
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return MaxminDQN(mdp_info, self.policy, self.approximator, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=DQNNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_steps_per_fit=1, n_approximators=3, use_cuda=False):
        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            loss=F.smooth_l1_loss,
            use_cuda=use_cuda)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            n_approximators=n_approximators,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)
