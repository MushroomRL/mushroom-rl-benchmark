import torch.optim as optim

from mushroom_rl.algorithms.value import CategoricalDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import AtariFeatureNetwork, FeedForwardNetwork
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

from .dqn import DQNBuilder


class CategoricalDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self._prepare(mdp_info, (AtariFeatureNetwork, FeedForwardNetwork))
        return CategoricalDQN(mdp_info, self.policy, self.approximator_params, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=AtariFeatureNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, v_min=-10, v_max=10,
                n_atoms=51, use_cuda=False):
        policy = EpsGreedy(Parameter(1.0, backend='torch'), backend='torch')

        approximator_params = dict(
            network=network,
            n_features=n_features,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}})

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
