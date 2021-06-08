from mushroom_rl.algorithms.value import SARSALambda, QLambda
from mushroom_rl.utils.parameters import ExponentialParameter, Parameter

from .td_finite import TDFiniteBuilder


class TDTraceBuilder(TDFiniteBuilder):
    """
    Builder for TD algorithms with eligibility traces and finite states.

    """
    def __init__(self, learning_rate, epsilon, epsilon_test, lambda_coeff, trace):
        """
        Constructor.

        lambda_coeff ([float, Parameter]): eligibility trace coefficient;
        trace (str): type of eligibility trace to use.

        """
        super().__init__(learning_rate, epsilon, epsilon_test, lambda_coeff=lambda_coeff, trace=trace)

    @classmethod
    def default(cls, learning_rate=.9, epsilon=0.1, decay_lr=0., decay_eps=0., epsilon_test=0., lambda_coeff=0.9,
                trace='replacing', get_default_dict=False):
        if decay_eps == 0:
            epsilon = Parameter(value=epsilon)
        else:
            epsilon = ExponentialParameter(value=epsilon, exp=decay_eps)
        if decay_lr == 0:
            learning_rate = Parameter(value=learning_rate)
        else:
            learning_rate = ExponentialParameter(value=learning_rate, exp=decay_lr)
        defaults = locals()

        builder = cls(learning_rate, epsilon, epsilon_test, lambda_coeff, trace)

        if get_default_dict:
            return builder, defaults
        else:
            return builder


class SARSALambdaBuilder(TDTraceBuilder):
    alg_class = SARSALambda


class QLambdaBuilder(TDTraceBuilder):
    alg_class = QLambda