from mushroom_rl.algorithms.value import QLearning, DoubleQLearning, SARSA, SpeedyQLearning, WeightedQLearning
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import ExponentialParameter, Parameter

from mushroom_rl_benchmark.builders import AgentBuilder


class TDFiniteBuilder(AgentBuilder):
    """
    AgentBuilder for a generic TD algorithm (for finite states).

    """
    def __init__(self, learning_rate, epsilon, epsilon_test, **alg_params):
        """
        Constructor.

        Args:
            epsilon (Parameter): exploration coefficient for learning;
            epsilon_test (Parameter): exploration coefficient for test.

        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_test = epsilon_test
        self.alg_params = alg_params

        super().__init__(n_steps_per_fit=1, compute_policy_entropy=False)

    def build(self, mdp_info):
        policy = EpsGreedy(self.epsilon)

        return self.alg_class(mdp_info, policy, self.learning_rate, **self.alg_params)

    def compute_Q(self, agent, states):
        q_max = agent.Q[states, :].max()

        return q_max

    def set_eval_mode(self, agent, eval):
        if eval:
            agent.policy.set_epsilon(self.epsilon_test)
        else:
            agent.policy.set_epsilon(self.epsilon)

    @classmethod
    def default(cls, learning_rate=.9, epsilon=0.1, decay_lr=0., decay_eps=0., epsilon_test=0., get_default_dict=False):
        if decay_eps == 0:
            epsilon = Parameter(value=epsilon)
        else:
            epsilon = ExponentialParameter(value=epsilon, exp=decay_eps)
        if decay_lr == 0:
            learning_rate = Parameter(value=learning_rate)
        else:
            learning_rate = ExponentialParameter(value=learning_rate, exp=decay_lr)
        defaults = locals()

        builder = cls(learning_rate, epsilon, epsilon_test)

        if get_default_dict:
            return builder, defaults
        else:
            return builder


class QLearningBuilder(TDFiniteBuilder):
    alg_class = QLearning

    def __init__(self, learning_rate, epsilon, epsilon_test):
        super().__init__(learning_rate, epsilon, epsilon_test)


class SARSABuilder(TDFiniteBuilder):
    alg_class = SARSA

    def __init__(self, learning_rate, epsilon, epsilon_test):
        super().__init__(learning_rate, epsilon, epsilon_test)


class SpeedyQLearningBuilder(TDFiniteBuilder):
    alg_class = SpeedyQLearning

    def __init__(self, learning_rate, epsilon, epsilon_test):
        super().__init__(learning_rate, epsilon, epsilon_test)


class DoubleQLearningBuilder(TDFiniteBuilder):
    alg_class = DoubleQLearning

    def __init__(self, learning_rate, epsilon, epsilon_test):
        super().__init__(learning_rate, epsilon, epsilon_test)

    def compute_Q(self, agent, states):
        q_max_0 = agent.Q[0][states, :].max()
        q_max_1 = agent.Q[1][states, :].max()

        q_max = (q_max_0 + q_max_1) / 2

        return q_max


class WeightedQLearningBuilder(TDFiniteBuilder):
    alg_class = WeightedQLearning

    def __init__(self, learning_rate, epsilon, epsilon_test, sampling, precision):
        """
        Constructor.

        Args:
            sampling (bool, True): use the approximated version to speed up
                the computation;
            precision (int, 1000): number of samples to use in the approximated
                version.

        """
        super().__init__(learning_rate, epsilon, epsilon_test, sampling=sampling, precision=precision)

    @classmethod
    def default(cls, learning_rate=.9, epsilon=0.1, decay_lr=0., decay_eps=0., epsilon_test=0., sampling=True,
                precision=1000,
                get_default_dict=False):
        if decay_eps == 0:
            epsilon = Parameter(value=epsilon)
        else:
            epsilon = ExponentialParameter(value=epsilon, exp=decay_eps)
        if decay_lr == 0:
            learning_rate = Parameter(value=learning_rate)
        else:
            learning_rate = ExponentialParameter(value=learning_rate, exp=decay_lr)
        defaults = locals()

        builder = cls(learning_rate, epsilon, epsilon_test, sampling, precision)

        if get_default_dict:
            return builder, defaults
        else:
            return builder
