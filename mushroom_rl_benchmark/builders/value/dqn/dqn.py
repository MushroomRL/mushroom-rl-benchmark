import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import AtariNetwork, QNetwork
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter

from mushroom_rl_benchmark.builders import AgentBuilder


class DQNBuilder(AgentBuilder):
    """
    AgentBuilder for Deep Q-Network (DQN).

    """

    algorithm = DQN

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
        super().__init__(n_steps_per_fit=n_steps_per_fit, compute_policy_entropy=False)

    def _prepare(self, mdp_info, feature_network=None):
        observation_shape = mdp_info.observation_space.shape
        is_atari = len(observation_shape) == 2
        history_length = self.alg_params.setdefault('history_length', 4 if is_atari else 1)

        if feature_network is not None:
            self.approximator_params['network'] = feature_network[0] if is_atari else feature_network[1]
        elif self.approximator_params['network'] is AtariNetwork and not is_atari:
            self.approximator_params['network'] = QNetwork

        self.approximator_params['input_shape'] = (history_length,) + observation_shape \
            if is_atari else observation_shape
        self.approximator_params['output_shape'] = (mdp_info.action_space.n,)
        self.approximator_params['n_actions'] = mdp_info.action_space.n

        self.epsilon = LinearParameter(1.0, threshold_value=0.05, n=1_000_000, backend='torch')
        self.epsilon_test = Parameter(0.01, backend='torch')
        self.policy = EpsGreedy(self.epsilon, backend='torch')

    def build(self, mdp_info):
        self._prepare(mdp_info)
        return self.algorithm(mdp_info, self.policy, self.approximator,
                              approximator_params=self.approximator_params, **self.alg_params)

    def compute_Q(self, agent, states):
        return agent.approximator(states).max(dim=-1).values.mean()

    def set_eval_mode(self, agent, eval):
        agent.policy.set_epsilon(self.epsilon_test if eval else self.epsilon)

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
            target_update_frequency=target_update_frequency)
        return cls(policy, TorchApproximator, approximator_params, alg_params, n_steps_per_fit)
