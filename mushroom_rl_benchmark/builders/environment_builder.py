from copy import deepcopy
from mushroom_rl.core import Environment

from mushroom_rl.environments import Atari


class EnvironmentBuilder:
    """
    Class to spawn instances of a MushroomRL environment

    """
    def __init__(self, env_name, env_params):
        """
        Constructor

        Args:
            env_name: name of the environment to build;
            env_params: required parameters to build the specified environment.

        """
        self.env_name = env_name
        self.env_params = env_params

    def build(self):
        """
        Build and return an environment

        """
        return Environment.make(self.env_name, **self.env_params)

    @staticmethod
    def set_eval_mode(env, eval):
        """
        Make changes to the environment for evaluation mode.

        Args:
            env (Environment): the environment to change;
            eval (bool): flag for activating evaluation mode.

        """
        if isinstance(env, Atari):
            if eval:
                env.set_episode_end(False)
            else:
                env.set_episode_end(True)

    def copy(self):
        """
        Create a deepcopy of the environment_builder and return it

        """
        return deepcopy(self)
