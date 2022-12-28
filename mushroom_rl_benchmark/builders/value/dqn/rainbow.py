import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.value import Rainbow
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl_benchmark.builders.network import DQNFeatureNetwork

from .dqn import DQNBuilder


class RainbowBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.epsilon = LinearParameter(value=0., threshold_value=0., n=1)
        self.epsilon_test = Parameter(value=0.)

        return Rainbow(mdp_info, self.policy, self.approximator_params, **self.alg_params)

    @classmethod
    def default(cls, lr=.0001, network=DQNFeatureNetwork, initial_replay_size=50000, max_replay_size=1000000,
                batch_size=32, target_update_frequency=2500, n_features=512, n_steps_per_fit=1, v_min=-10, v_max=10,
                n_atoms=51, n_steps_return=3, alpha_coeff=.5, use_cuda=False):
        policy = EpsGreedy(epsilon=Parameter(value=1.))

        approximator_params = dict(
            network=network,
            n_features=n_features,
            optimizer={
                'class': optim.Adam,
                'params': {'lr': lr}},
            use_cuda=use_cuda)

        beta = LinearParameter(.4, threshold_value=1, n=50000000 // 4)

        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            alpha_coeff=alpha_coeff,
            beta=beta,
            n_steps_return=n_steps_return,
            target_update_frequency=target_update_frequency
        )

        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)
