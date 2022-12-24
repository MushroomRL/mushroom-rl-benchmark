import sys
import torch
import numpy as np
from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J, parse_dataset, get_init_states

import mushroom_rl_benchmark.builders
from mushroom_rl_benchmark.builders import EnvironmentBuilder


class BenchmarkExperiment:
    def __init__(self, agent, env, quiet, n_epochs, n_steps=None, n_episodes=None, n_steps_test=None,
                 n_episodes_test=None, **kwargs):
        agent_builder_factory = getattr(mushroom_rl_benchmark.builders, f'{agent}Builder')
        self._agent_builder = agent_builder_factory.default(**kwargs)
        self._env_builder = EnvironmentBuilder(env, dict())  # TODO FIXME

        self._learn_params = dict(
            render=False,
            quiet=quiet
        )

        if n_steps is None and n_episodes is not None:
            self._learn_params['n_episodes'] = n_episodes
        elif n_steps is not None and n_episodes is None:
            self._learn_params['n_steps'] = n_steps
        else:
            raise AttributeError('Set parameter n_steps or n_episodes')

        self._eval_params = dict(
            render=False,
            quiet=quiet)

        if n_steps_test is None and n_episodes_test is not None:
            self._eval_params['n_episodes'] = n_episodes_test
        elif n_steps_test is not None and n_episodes_test is None:
            self._eval_params['n_steps'] = n_steps_test
        else:
            raise AttributeError('Set parameter n_steps_test or n_episodes_test')

        self._n_epochs = n_epochs

    def run(self, save_agent, quiet, seed, results_dir):

        np.random.seed(seed)
        torch.manual_seed(seed)

        mdp = self._env_builder.build()
        if hasattr(mdp, 'env'):
            mdp.env.seed(seed)
        agent = self._agent_builder.build(mdp.info)

        logger = Logger(agent.__class__.__name__, use_timestamp=True, seed=seed, results_dir=results_dir)
        core = Core(agent, mdp)

        if not quiet:
            logger.strong_line()
            logger.info('Starting experiment with seed {}'.format(seed))
            logger.strong_line()

        results_dict = self._evaluate_agent(core, self._eval_params, self._agent_builder, self._env_builder)

        if save_agent:
            logger.log_best_agent(agent, results_dict['J'])

        if not quiet:
            logger.epoch_info(0, **results_dict)

        for epoch in trange(self._n_epochs, disable=quiet, leave=False):
            try:
                core.learn(**self._learn_params, **self._agent_builder.get_fit_params())
            except:
                e = sys.exc_info()
                logger.error(f'EXECUTION FAILED: Epoch {epoch+1} SEED {seed}')
                logger.exception(e)
                sys.exit()

            results_dict = self._evaluate_agent(core, self._eval_params, self._agent_builder, self._env_builder)

            logger.log_numpy(**results_dict)

            if save_agent:
                logger.log_best_agent(agent, results_dict['J'])

            if not quiet:
                logger.epoch_info(epoch+1, **results_dict)

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
        env_builder.set_eval_mode(core.mdp, True)
        dataset = core.evaluate(**eval_params)
        agent_builder.set_eval_mode(core.agent, False)
        env_builder.set_eval_mode(core.mdp, False)

        # Compute J
        J = np.mean(compute_J(dataset, core.mdp.info.gamma))
        R = np.mean(compute_J(dataset))

        results_dict = dict(J=J, R=R)

        # Compute V
        if agent_builder.compute_value_function:
            states = get_init_states(dataset)
            V = agent_builder.compute_Q(
                agent=core.agent,
                states=states)
            results_dict['V'] = V

        # Compute Policy Entropy
        if agent_builder.compute_policy_entropy:
            if agent_builder.compute_entropy_with_states:
                E = core.agent.policy.entropy(parse_dataset(dataset)[0])
            else:
                E = core.agent.policy.entropy()
            results_dict['E'] = E

        return results_dict
