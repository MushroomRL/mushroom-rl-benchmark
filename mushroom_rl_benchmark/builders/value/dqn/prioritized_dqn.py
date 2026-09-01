import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import AtariNetwork
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory

from .dqn import DQNBuilder


class PrioritizedDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.alg_params['replay_memory'] = {
            'class': PrioritizedReplayMemory,
            'params': {
                'alpha': .6,
                'beta': LinearParameter(.4, threshold_value=1., n=25000000 // 4, backend='torch'),
            },
        }
        return super().build(mdp_info)

    @classmethod
    def default(cls, lr=.0001, network=AtariNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, use_cuda=False):
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
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)
