from mushroom_rl.algorithms.value import SARSALambdaContinuous
from mushroom_rl.policy import EpsGreedy

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.utils.parameters import ExponentialParameter, Parameter

from .td_continuous import TDContinuousBuilder


class SarsaLambdaContinuousBuilder(TDContinuousBuilder):
    """
    AgentBuilder for Sarsa(Lambda) Continuous. Using tiles as function approximator.

    """
    def __init__(self, policy, approximator, learning_rate, lambda_coeff, epsilon, epsilon_test,
                 n_tilings, n_tiles):
        """
        Constructor.

        Args:
            approximator (class): Q-function approximator.

        """
        self.approximator = approximator

        super().__init__(policy, learning_rate, lambda_coeff, epsilon, epsilon_test, n_tilings, n_tiles)

    def build(self, mdp_info):
        features, approximator_params = self._build_function_approximation(mdp_info)

        return SARSALambdaContinuous(mdp_info, self.policy, self.approximator, self.learning_rate, self.lambda_coeff,
                                     features, approximator_params)

    @classmethod
    def default(cls, alpha=.1, lambda_coeff=0.9, epsilon=0., decay_eps=0., epsilon_test=0., n_tilings=10, n_tiles=10):
        if decay_eps == 0:
            epsilon_p = Parameter(value=epsilon)
        else:
            epsilon_p = ExponentialParameter(value=epsilon, exp=decay_eps)

        epsilon_test_p = Parameter(value=epsilon_test)
        policy = EpsGreedy(epsilon=epsilon_p)

        lambda_coeff_p = Parameter(lambda_coeff)
        learning_rate = Parameter(alpha / n_tilings)

        return cls(policy, LinearApproximator, learning_rate, lambda_coeff_p, epsilon_p, epsilon_test_p,
                   n_tilings=n_tilings, n_tiles=n_tiles)
