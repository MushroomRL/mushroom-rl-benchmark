import numpy as np

from mushroom_rl.algorithms.policy_search import PGPE, RWR, REPS, ConstrainedREPS
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from mushroom_rl_benchmark.builders import AgentBuilder


class BBOBuilder(AgentBuilder):
    """
    AgentBuilder for Black Box optimization methods.
    The current builder uses a simple deterministic linear policy and gaussian Diagonal distribution.

    """
    def __init__(self, n_episodes_per_fit, **kwargs):
        """
        Constructor.

        Args:
            optimizer (Optimizer): optimizer to be used by the policy gradient algorithm;
            **kwargs: others algorithms parameters.

        """

        self.algorithm_params = kwargs

        super().__init__(n_episodes_per_fit=n_episodes_per_fit, compute_policy_entropy=False,
                         compute_value_function=False)

    def _build(self, mdp_info):
        approximator = Regressor(LinearApproximator,
                                 input_shape=mdp_info.observation_space.shape,
                                 output_shape=mdp_info.action_space.shape)

        n_weights = approximator.weights_size
        mu = np.zeros(n_weights)
        sigma = 2e-0 * np.ones(n_weights)
        policy = DeterministicPolicy(approximator)
        dist = GaussianDiagonalDistribution(mu, sigma)

        return self.alg_class(mdp_info, dist, policy, **self.algorithm_params)

    @classmethod
    def default(cls, n_episodes_per_fit=25, alpha=1.0e-2):
        raise NotImplementedError

    def compute_Q(self, agent, states):
        pass


class PGPEBuilder(BBOBuilder):
    alg_class = PGPE

    def __init__(self, n_episodes_per_fit, optimizer):
        super().__init__(n_episodes_per_fit, optimizer=optimizer)

    @classmethod
    def default(cls, n_episodes_per_fit=25, alpha=3e-1):
        optimizer = AdaptiveOptimizer(alpha)
        return cls(n_episodes_per_fit, optimizer)



class RWRBuilder(BBOBuilder):
    alg_class = RWR

    def __init__(self, n_episodes_per_fit, beta):
        super().__init__(n_episodes_per_fit, beta=beta)

    @classmethod
    def default(cls, n_episodes_per_fit=25, beta=1e-2):
        return cls(n_episodes_per_fit, beta)


class REPSBuilder(BBOBuilder):
    alg_class = REPS

    def __init__(self, n_episodes_per_fit, eps):
        super().__init__(n_episodes_per_fit, eps=eps)

    @classmethod
    def default(cls, n_episodes_per_fit=25, eps=5e-2):

        return cls(n_episodes_per_fit, eps)


class ConstrainedREPSBuilder(BBOBuilder):
    alg_class = ConstrainedREPS

    def __init__(self, n_episodes_per_fit, eps, kappa):
        super().__init__(n_episodes_per_fit, eps=eps, kappa=kappa)

    @classmethod
    def default(cls, n_episodes_per_fit=25, eps=5e-2, kappa=1e-2):
        return cls(n_episodes_per_fit, eps, kappa)
