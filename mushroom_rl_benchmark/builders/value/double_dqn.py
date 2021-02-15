from mushroom_rl.algorithms.value import DoubleDQN
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from .dqn import DQNBuilder


class DoubleDQNBuilder(DQNBuilder):
    def build(self, mdp_info):
        self.approximator_params['input_shape'] = mdp_info.observation_space.shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n
        self.epsilon = LinearParameter(value=1, threshold_value=.05, n=1000000)
        self.epsilon_test = Parameter(value=.01)

        return DoubleDQN(mdp_info, self.policy, self.approximator, self.approximator_params, **self.alg_params)