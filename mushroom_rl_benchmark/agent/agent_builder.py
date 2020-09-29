from copy import deepcopy

class AgentBuilder:

    def __init__(self, n_steps_per_fit, compute_policy_entropy=True, compute_entropy_with_states=False, preprocessors=None):
        """
        Initialize AgentBuilder
        """
        self.set_n_steps_per_fit(n_steps_per_fit)
        self.set_preprocessors(preprocessors)
        self.compute_policy_entropy = compute_policy_entropy
        self.compute_entropy_with_states = compute_entropy_with_states

    def set_n_steps_per_fit(self, n_steps_per_fit):
        """
        Set n_steps_per_fit for the specific agent

        Args:
            n_steps_per_fit: number of steps per fit
        """
        self._n_steps_per_fit = n_steps_per_fit

    def get_n_steps_per_fit(self):
        """
        Get n_steps_per_fit for the specific agent
        """
        return self._n_steps_per_fit

    def set_preprocessors(self, preprocessors):
        """
        Set preprocessor for the specific agent

        Args:
            preprocessors: list of preprocessor classes
        """
        self._preprocessors = preprocessors if preprocessors is not None else list()

    def get_preprocessors(self):
        """
        Get preprocessors for the specific agent
        """
        return self._preprocessors

    def copy(self):
        """
        Create a deepcopy of the agent_builder and return it
        """
        return deepcopy(self)

    def build(self):
        """
        Build and return the agent
        """
        raise NotImplementedError('AgentBuilder is an abstract class')

    def compute_Q(self, agent, states):
        """
        Compute the Q Value for an agent
        """
        raise NotImplementedError('AgentBuilder is an abstract class')
    
    @classmethod
    def default(cls):
        """
        Create a default initialization for the specific agent and return a new AgentBuilder class
        """
        raise NotImplementedError('AgentBuilder is an abstract class')