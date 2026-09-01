from mushroom_rl.core import Environment

from mushroom_rl_benchmark.builders import EnvironmentBuilder


def test_environment_builder_uses_registered_environments():
    builder = EnvironmentBuilder('GridWorld', {'height': 5, 'width': 6, 'goal': [3, 4]})

    environment = builder.build()

    assert environment.info.observation_space.n == 30
    assert 'GridWorld' in Environment.list_registered()
