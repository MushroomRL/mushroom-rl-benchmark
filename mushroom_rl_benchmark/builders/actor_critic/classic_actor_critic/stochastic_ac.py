import numpy as np

from mushroom_rl_benchmark.builders import AgentBuilder

from mushroom_rl.algorithms.actor_critic import StochasticAC
from mushroom_rl.policy import StateLogStdGaussianPolicy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.utils.parameters import Parameter


class StochasticACBuilder(AgentBuilder):
    """
    Builder for the stochastic actor critic algorithm.
    Using linear approximator with tiles for mean, standard deviation and value function approximator.
    The value function approximator also uses a bias term.

    """
    def __init__(self, std_0, alpha_theta, alpha_v, lambda_par, n_tilings, n_tiles, **kwargs):
        """
        Constructor.

        Args:
            std_0 (float): initial standard deviation;
            alpha_theta (Parameter): Learning rate for the policy;
            alpha_v (Parameter): Learning rate for the value function;
            n_tilings (int): number of tilings to be used as approximator;
            n_tiles (int): number of tiles for each state space dimension.

        """
        self._std_0 = std_0
        self._alpha_theta = alpha_theta
        self._alpha_v = alpha_v
        self._n_tilings = n_tilings
        self._n_tiles = n_tiles
        self._lambda_par = lambda_par

        super().__init__(n_steps_per_fit=1, compute_policy_entropy=False)

    def build(self, mdp_info):
        tilings = Tiles.generate(self._n_tilings, [self._n_tiles]*mdp_info.observation_space.shape[0],
                                 mdp_info.observation_space.low, mdp_info.observation_space.high)

        phi = Features(tilings=tilings)

        tilings_v = tilings + Tiles.generate(1, [1, 1], mdp_info.observation_space.low, mdp_info.observation_space.high)

        psi = Features(tilings=tilings_v)

        input_shape = (phi.size,)

        mu = Regressor(LinearApproximator, input_shape=input_shape,
                       output_shape=mdp_info.action_space.shape)

        std = Regressor(LinearApproximator, input_shape=input_shape,
                        output_shape=mdp_info.action_space.shape)

        std.set_weights(np.log(self._std_0) / self._n_tilings * np.ones(std.weights_size))

        policy = StateLogStdGaussianPolicy(mu, std)

        return StochasticAC(mdp_info, policy, self._alpha_theta, self._alpha_v, lambda_par=.9,
                            value_function_features=psi, policy_features=phi)

    @classmethod
    def default(cls, std_0=1.0, alpha_theta=1e-3, alpha_v=1e-1, lambda_par=0.9, n_tilings=10, n_tiles=11,
                get_default_dict=False):
        defaults = locals()

        alpha_theta_p = Parameter(alpha_theta / n_tilings)
        alpha_v_p = Parameter(alpha_v / n_tilings)

        builder = cls(std_0, alpha_theta_p, alpha_v_p, lambda_par, n_tilings, n_tiles)

        if get_default_dict:
            return builder, defaults
        else:
            return builder

    def compute_Q(self, agent, states):
        return agent._V(agent._psi(states)).mean()
