import numpy as np

from mushroom_rl_benchmark.builders import AgentBuilder

from mushroom_rl.algorithms.actor_critic import COPDAC_Q
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.utils.parameters import Parameter


class COPDAC_QBuilder(AgentBuilder):
    """
    Builder for the COPDAQ_Q actor critic algorithm.
    Using linear approximator with tiles for the mean and value function approximator.

    """
    def __init__(self, std_exp, std_eval, alpha_theta, alpha_omega, alpha_v, n_tilings, n_tiles, **kwargs):
        """
        Constructor.

        Args:
            std_exp (float): exploration standard deviation;
            std_eval (float): evaluation standard deviation;
            alpha_theta (Parameter): Learning rate for the policy;
            alpha_omega (Parameter): Learning rate for the
            alpha_v (Parameter): Learning rate for the value function;
            n_tilings (int): number of tilings to be used as approximator;
            n_tiles (int): number of tiles for each state space dimension.

        """
        self._std_exp = std_exp
        self._std_eval = std_eval
        self._alpha_theta = alpha_theta
        self._alpha_omega = alpha_omega
        self._alpha_v = alpha_v
        self._n_tilings = n_tilings
        self._n_tiles = n_tiles

        self._sigma_exp = None
        self._sigma_eval = None

        super().__init__(n_steps_per_fit=1, compute_policy_entropy=False)

    def _build(self, mdp_info):
        tilings = Tiles.generate(self._n_tilings, [self._n_tiles] * mdp_info.observation_space.shape[0],
                                 mdp_info.observation_space.low, mdp_info.observation_space.high)

        phi = Features(tilings=tilings)

        input_shape = (phi.size,)

        mu = Regressor(LinearApproximator, input_shape=input_shape,
                       output_shape=mdp_info.action_space.shape)

        self._sigma_exp = self._std_exp * np.eye(mdp_info.action_space.shape[0])
        self._sigma_eval = self._std_eval * np.eye(mdp_info.action_space.shape[0])

        policy = GaussianPolicy(mu, self._sigma_exp)

        return COPDAC_Q(mdp_info, policy, mu, self._alpha_theta, self._alpha_omega, self._alpha_v,
                        value_function_features=phi, policy_features=phi)

    def set_eval_mode(self, agent, eval):
        if eval:
            agent.policy.set_sigma(self._sigma_eval)
        else:
            agent.policy.set_sigma(self._sigma_exp)

    @classmethod
    def default(cls, std_exp=1e-1, std_eval=1e-3, alpha_theta=5e-3, alpha_omega=5e-1, alpha_v=5e-1,
                n_tilings=10, n_tiles=11):

        alpha_theta_p = Parameter(alpha_theta / n_tilings)
        alpha_omega_p = Parameter(alpha_omega / n_tilings)
        alpha_v_p = Parameter(alpha_v / n_tilings)

        return cls(std_exp, std_eval, alpha_theta_p, alpha_omega_p, alpha_v_p, n_tilings, n_tiles)

    def compute_Q(self, agent, states):
        return agent._V(agent._psi(states)).mean()
