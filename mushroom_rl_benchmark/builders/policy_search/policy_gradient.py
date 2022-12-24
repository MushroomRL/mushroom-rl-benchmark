import numpy as np

from mushroom_rl.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from mushroom_rl_benchmark.builders import AgentBuilder


class PolicyGradientBuilder(AgentBuilder):
    """
    AgentBuilder for Policy Gradient Methods.
    The current builder uses a state dependant gaussian with diagonal standard deviation and linear mean.

    """
    def __init__(self, n_episodes_per_fit, optimizer, **kwargs):
        """
        Constructor.

        Args:
            optimizer (Optimizer): optimizer to be used by the policy gradient algorithm;
            **kwargs: others algorithms parameters.

        """

        self.algorithm_params = dict(optimizer=optimizer)
        self.algorithm_params.update(**kwargs)

        super().__init__(n_episodes_per_fit=n_episodes_per_fit, compute_policy_entropy=False,
                         compute_value_function=False)

    def _build(self, mdp_info):
        mu = Regressor(LinearApproximator,
                       input_shape=mdp_info.observation_space.shape,
                       output_shape=mdp_info.action_space.shape)

        sigma = Regressor(LinearApproximator,
                          input_shape=mdp_info.observation_space.shape,
                          output_shape=mdp_info.action_space.shape)

        sigma_weights = .25 * np.ones(sigma.weights_size)
        sigma.set_weights(sigma_weights)

        policy = StateStdGaussianPolicy(mu, sigma)

        return self.alg_class(mdp_info, policy, **self.algorithm_params)

    @classmethod
    def default(cls, n_episodes_per_fit=25, alpha=1.0e-2, get_default_dict=False):
        defaults = locals()

        optimizer = AdaptiveOptimizer(eps=alpha)
        builder = cls(n_episodes_per_fit, optimizer)

        if get_default_dict:
            return builder, defaults
        else:
            return builder

    def compute_Q(self, agent, states):
        pass


class REINFORCEBuilder(PolicyGradientBuilder):
    alg_class = REINFORCE


class GPOMDPBuilder(PolicyGradientBuilder):
    alg_class = GPOMDP


class eNACBuilder(PolicyGradientBuilder):
    alg_class = eNAC