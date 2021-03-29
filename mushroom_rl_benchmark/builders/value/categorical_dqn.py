import torch
import torch.optim as optim

from mushroom_rl.algorithms.value import CategoricalDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl_benchmark.builders.network import DQNFeatureNetwork

from .dqn import DQNBuilder


class CategoricalDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.approximator_params['network'] = DQNFeatureNetwork
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return CategoricalDQN(mdp_info, self.policy, self.approximator_params, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=DQNFeatureNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, v_min=-10, v_max=10,
                n_atoms=51, use_cuda=False, get_default_dict=False):
        defaults = locals()

        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            n_features=n_features,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
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

        builder = cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)

        if get_default_dict:
            return builder, defaults
        else:
            return builder
