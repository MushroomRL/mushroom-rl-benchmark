from .dqn import DQNBuilder
from .double_dqn import DoubleDQNBuilder
from .prioritized_dqn import PrioritizedDQNBuilder
from .averaged_dqn import AveragedDQNBuilder
from .dueling_dqn import DuelingDQNBuilder
from .maxmin_dqn import MaxminDQNBuilder
from .categorical_dqn import CategoricalDQNBuilder
from .rainbow import RainbowBuilder


__all__ = [
    'DQNBuilder',
    'DoubleDQNBuilder',
    'PrioritizedDQNBuilder',
    'AveragedDQNBuilder',
    'DuelingDQNBuilder',
    'MaxminDQNBuilder',
    'CategoricalDQNBuilder',
    'RainbowBuilder'
]

try:
    from .noisy_dqn import NoisyDQNBuilder
    __all__ += ['NoisyDQNBuilder']
except ImportError:
    pass

