from mushroom_rl.algorithms.value import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter

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
        self.learning_rate = Parameter(learning_rate)
        self.epsilon = epsilon
        self.epsilon_test = epsilon_test
        self.alg_params = alg_params

        super().__init__(1, compute_policy_entropy=False)

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
    def default(cls, learning_rate=.9, epsilon=0.1, epsilon_test=0., get_default_dict=False):
        defaults = locals()

        builder = cls(learning_rate, epsilon, epsilon_test)

        if get_default_dict:
            return builder, defaults
        else:
            return builder


class QLearningBuilder(TDFiniteBuilder):
    alg_class = QLearning

    def __init__(self, learning_rate, epsilon, epsilon_test):
        """
        Constructor.


        """
        super().__init__(learning_rate, epsilon, epsilon_test)


class SARSABuilder(TDFiniteBuilder):
    alg_class = SARSA

    def __init__(self, learning_rate, epsilon, epsilon_test):
        """
        Constructor.


        """
        super().__init__(learning_rate, epsilon, epsilon_test)


class SARSALambdaBuilder(TDFiniteBuilder):
    alg_class = SARSALambda

    def __init__(self, learning_rate, epsilon, epsilon_test, lambda_coeff, trace='replacing'):
        """
        Constructor.

        lambda_coeff ([float, Parameter]): eligibility trace coefficient;
        trace (str): type of eligibility trace to use.

        """
        super().__init__(learning_rate, epsilon, epsilon_test, lambda_coeff=lambda_coeff, trace=trace)

    @classmethod
    def default(cls, learning_rate=.9, epsilon=0.1, epsilon_test=0., lambda_coeff=0.9, trace='replacing',
                get_default_dict=False):
        defaults = locals()

        builder = cls(learning_rate, epsilon, epsilon_test, lambda_coeff, trace)

        if get_default_dict:
            return builder, defaults
        else:
            return builder


class DoubleQLearningBuilder(TDFiniteBuilder):
    alg_class = DoubleQLearning

    def __init__(self, learning_rate, epsilon, epsilon_test):
        """
        Constructor.


        """
        super().__init__(learning_rate, epsilon, epsilon_test)


class SpeedyQLearningBuilder(TDFiniteBuilder):
    alg_class = SpeedyQLearning

    def __init__(self, learning_rate, epsilon, epsilon_test):
        """
        Constructor.


        """
        super().__init__(learning_rate, epsilon, epsilon_test)


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
    def default(cls, learning_rate=.9, epsilon=0.1, epsilon_test=0., sampling=True, precision=1000,
                get_default_dict=False):
        defaults = locals()

        builder = cls(learning_rate, epsilon, epsilon_test, sampling, precision)

        if get_default_dict:
            return builder, defaults
        else:
            return builder