import torch
import numpy as np
from pathlib import Path
from tqdm import trange

from mushroom_rl.core import Core, Environment, Logger
from mushroom_rl.utils.torch_utils import TorchUtils

import mushroom_rl_benchmark.builders
from mushroom_rl_benchmark.builders import EnvironmentBuilder


class BenchmarkExperiment:
    def __init__(self, agent_name, env_name, agent_params, env_params, results_dir, n_epochs,
                 n_steps=None, n_episodes=None, n_steps_test=None, n_episodes_test=None,
                 use_cuda=False, log_console=True):
        self._agent_name = agent_name
        agent_builder_factory = getattr(mushroom_rl_benchmark.builders, f'{agent_name}Builder')
        agent_params = dict(agent_params)
        use_cuda = agent_params.pop('use_cuda', use_cuda)
        self._agent_builder = agent_builder_factory.default(**agent_params)
        self._env_builder = EnvironmentBuilder(env_name, env_params)
        self._results_dir = Path(results_dir)
        self._log_console = log_console

        self._learn_params = dict(render=False, quiet=not log_console)
        if n_steps is None and n_episodes is not None:
            self._learn_params['n_episodes'] = n_episodes
        elif n_steps is not None and n_episodes is None:
            self._learn_params['n_steps'] = n_steps
        else:
            raise AttributeError('Set parameter n_steps or n_episodes')

        self._eval_params = dict(render=False, quiet=not log_console)
        if n_steps_test is None and n_episodes_test is not None:
            self._eval_params['n_episodes'] = n_episodes_test
        elif n_steps_test is not None and n_episodes_test is None:
            self._eval_params['n_steps'] = n_steps_test
        else:
            raise AttributeError('Set parameter n_steps_test or n_episodes_test')

        self._n_epochs = n_epochs
        self._use_cuda = use_cuda

    @property
    def path(self):
        return self._results_dir / self._agent_name

    def run(self, save_agent=False, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self._use_cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested, but it is not available')
        TorchUtils.set_default_device('cuda:0' if self._use_cuda else 'cpu')

        mdp = self._env_builder.build()
        if type(mdp).seed is not Environment.seed:
            mdp.seed(seed)
        agent = self._agent_builder.build(mdp.info)
        for preprocessor in self._agent_builder.get_preprocessors():
            agent.add_core_preprocessor(preprocessor(mdp.info))

        logger_name = f'{self._results_dir.name}/{self._agent_name}'
        logger = Logger(logger_name, results_dir=self._results_dir.parent,
                        log_console=self._log_console, log_file_name=self._agent_name, seed=seed)
        logger.log_experiment_info(agent, mdp, seed=seed)
        core = Core(agent, mdp, logger=logger)

        results_dict = self._evaluate_agent(core, self._eval_params, self._agent_builder, self._env_builder)
        logger.log_evaluation(0, **results_dict)

        if save_agent:
            logger.log_best_agent(agent, results_dict['J'])

        for epoch in trange(self._n_epochs, disable=not self._log_console, leave=False):
            core.learn(**self._learn_params, **self._agent_builder.get_fit_params())
            results_dict = self._evaluate_agent(core, self._eval_params, self._agent_builder, self._env_builder)
            logger.log_evaluation(epoch + 1, **results_dict)

            if save_agent:
                logger.log_best_agent(agent, results_dict['J'])

    @staticmethod
    def _evaluate_agent(core, eval_params, agent_builder, env_builder):
        """
        Function to compute the metrics.

        Args:
            eval_params (dict): parameters for running the evaluation;
            agent_builder (AgentBuilder): the agent builder;
            env_builder (EnvironmentBuilder): environment builder to spawn an environment;

        """
        agent_builder.set_eval_mode(core.agent, True)
        env_builder.set_eval_mode(core.env, True)
        dataset = core.evaluate(**eval_params)
        agent_builder.set_eval_mode(core.agent, False)
        env_builder.set_eval_mode(core.env, False)

        # Compute J
        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()

        results_dict = dict(J=J, R=R)

        # Compute V
        if agent_builder.compute_value_function:
            states = dataset.get_init_states()
            V = agent_builder.compute_Q(agent=core.agent, states=states)
            if isinstance(V, torch.Tensor):
                V = V.detach().item()
            results_dict['V'] = V

        # Compute Policy Entropy
        if agent_builder.compute_policy_entropy:
            if agent_builder.compute_entropy_with_states:
                states = core.agent.history_manager.parse_history(dataset)[0]
                entropy = core.agent.policy.entropy(states)
            else:
                entropy = core.agent.policy.entropy()
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.detach().item()
            results_dict['E'] = entropy

        return results_dict
