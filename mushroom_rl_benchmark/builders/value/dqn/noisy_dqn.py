import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.value import NoisyDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from mushroom_rl_benchmark.builders.network import DQNFeatureNetwork

from .dqn import DQNBuilder


class NoisyDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.epsilon = LinearParameter(value=0, threshold_value=0, n=1)
        self.epsilon_test = Parameter(value=0)

        return NoisyDQN(mdp_info, self.policy, self.approximator_params, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=DQNFeatureNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, use_cuda=False):
        policy = EpsGreedy(epsilon=Parameter(value=0.))

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
