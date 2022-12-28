from copy import deepcopy
import mushroom_rl.utils.preprocessors as m_prep


class AgentBuilder:
    """
    Base class to spawn instances of a MushroomRL agent

    """
    def __init__(self, n_steps_per_fit=None, n_episodes_per_fit=None,
                 compute_policy_entropy=True, compute_entropy_with_states=False,
                 compute_value_function=True, preprocessors=None):
        """
        Initialize AgentBuilder

        """
        assert (n_episodes_per_fit is None and n_steps_per_fit is not None) or \
               (n_episodes_per_fit is not None and n_steps_per_fit is None)

        self._preprocessors = None
        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit
        self._configure_preprocessors(preprocessors)
        self.compute_policy_entropy = compute_policy_entropy
        self.compute_entropy_with_states = compute_entropy_with_states
        self.compute_value_function = compute_value_function

    def get_fit_params(self):
        """
        Get n_steps_per_fit and n_episodes_per_fit for the specific AgentBuilder

        """
        return dict(n_steps_per_fit=self._n_steps_per_fit, n_episodes_per_fit=self._n_episodes_per_fit)

    def build(self, mdp_info):
        """
        Build and return the AgentBuilder

        Args:
            mdp_info (MDPInfo): information about the environment.

        """
        agent = self._build(mdp_info)
        self._set_preprocessors(agent, mdp_info)

        return agent

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

    def _build(self, mdp_info):
        raise NotImplementedError('AgentBuilder is an abstract class')

    def _set_preprocessors(self, agent, mdp_info):
        for p in self._preprocessors:
            agent.add_preprocessor(p(mdp_info))

    def _configure_preprocessors(self, preprocessors):
        """
        Configure the preprocessors for the specific AgentBuilder

        Args:
            preprocessors: list of preprocessor classes.

        """

        if preprocessors:
            preprocessors = preprocessors if isinstance(preprocessors, list) else [preprocessors]
            self._preprocessors = [getattr(m_prep, p) for p in preprocessors]
        else:
            self._preprocessors = list()
    
    @classmethod
    def default(cls, **kwargs):
        """
        Create a default initialization for the specific AgentBuilder and return it

        """
        raise NotImplementedError('AgentBuilder is an abstract class')


