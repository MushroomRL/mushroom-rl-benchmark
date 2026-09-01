import numpy as np
import pytest

from mushroom_rl.core import Box, Discrete, Environment, MDPInfo
from mushroom_rl_benchmark import builders


@pytest.mark.parametrize('name', ['A2C', 'PPO', 'TRPO', 'DDPG', 'TD3', 'SAC'])
def test_deep_actor_critic_builders(name):
    environment = Environment.make('InvertedPendulum', horizon=10)

    agent = getattr(builders, f'{name}Builder').default().build(environment.info)

    assert agent.mdp_info is environment.info


@pytest.mark.parametrize('name', ['REINFORCE', 'GPOMDP', 'eNAC', 'PGPE', 'RWR', 'REPS', 'ConstrainedREPS'])
def test_policy_search_builders(name):
    environment = Environment.make('LQR', dimensions=2)

    agent = getattr(builders, f'{name}Builder').default().build(environment.info)

    assert agent.mdp_info is environment.info


@pytest.mark.parametrize('name', ['StochasticAC', 'COPDAC_Q'])
def test_classic_actor_critic_builders(name):
    environment = Environment.make('InvertedPendulum', horizon=10)

    agent = getattr(builders, f'{name}Builder').default().build(environment.info)

    assert agent.mdp_info is environment.info


@pytest.mark.parametrize('name', ['DQN', 'DoubleDQN', 'PrioritizedDQN', 'AveragedDQN', 'DuelingDQN',
                                  'MaxminDQN', 'CategoricalDQN', 'NoisyDQN', 'Rainbow'])
def test_dqn_builders(name):
    mdp_info = MDPInfo(Box(np.zeros((84, 84)), np.full((84, 84), 255)), Discrete(4),
                       gamma=0.99, horizon=100)
    builder = getattr(builders, f'{name}Builder').default(initial_replay_size=2, max_replay_size=10,
                                                          batch_size=2)

    agent = builder.build(mdp_info)

    assert agent.mdp_info is mdp_info
