import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from mushroom_rl_benchmark.builders import AgentBuilder
from mushroom_rl_benchmark.builders.network import DQNNetwork


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
                batch_size=32, target_update_frequency=2500, n_steps_per_fit=1, use_cuda=False, get_default_dict=False):
        defaults = locals()
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

        builder = cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)

        if get_default_dict:
            return builder, defaults
        else:
            return builder
