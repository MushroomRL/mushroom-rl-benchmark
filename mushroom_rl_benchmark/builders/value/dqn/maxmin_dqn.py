import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.value import MaxminDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import AtariNetwork
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

from .dqn import DQNBuilder


class MaxminDQNBuilder(DQNBuilder):
    algorithm = MaxminDQN

    def build(self, mdp_info):
        return super().build(mdp_info)

    @classmethod
    def default(cls, lr=.0001, network=AtariNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, n_approximators=3,
                use_cuda=False):
        policy = EpsGreedy(Parameter(1.0, backend='torch'), backend='torch')

        approximator_params = dict(
            network=network,
            n_features=n_features,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            loss=F.smooth_l1_loss)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            n_approximators=n_approximators,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)
