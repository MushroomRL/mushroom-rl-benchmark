import numpy as np

from mushroom_rl_benchmark.builders import QLearningBuilder


class Agent:
    Q = np.array([[1.0, 2.0], [10.0, 0.0]])


def test_value_metric_averages_the_greedy_value_of_each_state():
    builder = QLearningBuilder.default()

    value = builder.compute_Q(Agent(), np.array([0, 1]))

    assert value == 6.0
