from copy import deepcopy
import mushroom_rl.environments

class EnvironmentBuilder:

    def __init__(self, env_name, env_params):
        """
        Initialize EnvironmentBuilder

        Args:
            env_name: name of the environment to build
            env_params: required parameters to build the specified environment
        """
        self.env_name = env_name
        self.env_params = env_params

    def build(self):
        """
        Build and return an environment
        """
        environment = getattr(mushroom_rl.environments, self.env_name)
        return environment(*self.env_params.values())

    def copy(self):
        """
        Create a deepcopy of the environment_builder and return it
        """
        return deepcopy(self)