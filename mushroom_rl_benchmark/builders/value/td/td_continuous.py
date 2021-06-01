from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.utils.parameters import Parameter

from mushroom_rl_benchmark.builders import AgentBuilder


class TDContinuousBuilder(AgentBuilder):
    """
    AgentBuilder for Sarsa(Lambda) Continuous. Using tiles as function approximator.

    """
    def __init__(self, policy, learning_rate, lambda_coeff, epsilon, epsilon_test, n_tilings, n_tiles):
        """
        Constructor.

        Args:
            policy (TDPolicy): policy class;
            approximator_params (dict): parameters for the approximator;
            lambda_coeff (Parameter): lambda coefficient for eligibility traces;
            epsilon (Parameter): exploration coefficient for learning;
            epsilon_test (Parameter): exploration coefficient for test;
            n_tilings (int): number of tilings to use;
            n_tiles (int): number of tiles for each dimension.

        """
        self.policy = policy
        self.learning_rate = learning_rate
        self.lambda_coeff = lambda_coeff
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.epsilon = epsilon
        self.epsilon_test = epsilon_test

        super().__init__(n_steps_per_fit=1, compute_policy_entropy=False)

    def _build_function_approximation(self, mdp_info):
        tilings = Tiles.generate(self.n_tilings, [self.n_tiles] * mdp_info.observation_space.shape[0],
                                 mdp_info.observation_space.low,
                                 mdp_info.observation_space.high)
        features = Features(tilings=tilings)

        approximator_params = dict(input_shape=(features.size,),
                                   output_shape=(mdp_info.action_space.n,),
                                   n_actions=mdp_info.action_space.n)

        return features, approximator_params

    def build(self, mdp_info):
        raise NotImplementedError

    def compute_Q(self, agent, states):
        q_max = agent.Q(agent.phi(states)).max()

        return q_max

    def set_eval_mode(self, agent, eval):
        if eval:
            agent.policy.set_epsilon(self.epsilon_test)
        else:
            agent.policy.set_epsilon(self.epsilon)

    @classmethod
    def default(cls, get_default_dict=False, **kwargs):
        raise NotImplementedError

