from copy import deepcopy
import mushroom_rl.utils.preprocessors as m_prep


class AgentBuilder:
    """
    Base class to spawn instances of a MushroomRL agent

    """
    def __init__(self, n_steps_per_fit, compute_policy_entropy=True, compute_entropy_with_states=False,
                 preprocessors=None):
        """
        Initialize AgentBuilder

        """
        self._preprocessors = None
        self.set_n_steps_per_fit(n_steps_per_fit)
        self.set_preprocessors(preprocessors)
        self.compute_policy_entropy = compute_policy_entropy
        self.compute_entropy_with_states = compute_entropy_with_states

    def set_n_steps_per_fit(self, n_steps_per_fit):
        """
        Set n_steps_per_fit for the specific AgentBuilder

        Args:
            n_steps_per_fit: number of steps per fit.

        """
        self._n_steps_per_fit = n_steps_per_fit

    def get_n_steps_per_fit(self):
        """
        Get n_steps_per_fit for the specific AgentBuilder

        """
        return self._n_steps_per_fit

    def set_preprocessors(self, preprocessors):
        """
        Set preprocessor for the specific AgentBuilder

        Args:
            preprocessors: list of preprocessor classes.

        """

        if preprocessors:
            preprocessors = preprocessors if isinstance(preprocessors, list) else [preprocessors]
            self._preprocessors = [getattr(m_prep, p) if isinstance(p, str)
                                   else p
                                   for p in preprocessors]
        else:
            self._preprocessors = list()

    def get_preprocessors(self):
        """
        Get preprocessors for the specific AgentBuilder

        """
        return self._preprocessors

    def copy(self):
        """
        Create a deepcopy of the AgentBuilder and return it

        """
        return deepcopy(self)

    def build(self, mdp_info):
        """
        Build and return the AgentBuilder

        Args:
            mdp_info (MDPInfo): information about the environment.

        """
        raise NotImplementedError('AgentBuilder is an abstract class')

    def compute_Q(self, agent, states):
        """
        Compute the Q Value for an AgentBuilder

        Args:
            agent (Agent): the considered agent;
            states (np.ndarray): the set of states over which we need
                to compute the Q function.

        """
        raise NotImplementedError('AgentBuilder is an abstract class')

    def set_eval_mode(self, agent, eval):
        """
        Set the eval mode for the agent.
        This function can be overwritten by any agent builder to setup
        specific evaluation mode for the agent.

        Args:
            agent (Agent): the considered agent;
            eval (bool): whether to set eval mode (true) or learn mode.

        """
        pass
    
    @classmethod
    def default(cls, get_default_dict=False, **kwargs):
        """
        Create a default initialization for the specific AgentBuilder and return it

        """
        raise NotImplementedError('AgentBuilder is an abstract class')
