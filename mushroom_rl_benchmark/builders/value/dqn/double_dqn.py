from mushroom_rl.algorithms.value import DoubleDQN

from .dqn import DQNBuilder


class DoubleDQNBuilder(DQNBuilder):
    algorithm = DoubleDQN
