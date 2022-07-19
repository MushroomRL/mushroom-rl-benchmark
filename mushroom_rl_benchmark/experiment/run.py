import sys
import torch
import numpy as np
from copy import deepcopy

from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J, parse_dataset, get_init_states

from tqdm import trange


def exec_run(agent_builder, env_builder, n_epochs, n_steps=None, n_episodes=None, n_steps_test=None,
             n_episodes_test=None, seed=None, save_agent=False, quiet=True, **kwargs):
    """
    Function that handles the execution of an experiment run.

    Args:
        agent_builder (AgentBuilder): agent builder to spawn an agent;
        env_builder (EnvironmentBuilder): environment builder to spawn an environment;
        n_epochs (int): number of epochs;
        n_steps (int, None): number of steps per epoch;
        n_episodes (int, None): number of episodes per epoch;
        n_steps_test (int, None): number of steps for testing;
        n_episodes_test (int, None): number of episodes for testing;
        seed (int, None): the seed;
        save_agent (bool, False): select if the agent should be logged or not;
        quiet (bool, True): select if run should print execution information.

    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    mdp = env_builder.build()
    if hasattr(mdp, 'env'):
        mdp.env.seed(seed)
    agent = agent_builder.build(mdp.info)
    preprocessors = [prepro(mdp.info) for prepro in agent_builder.get_preprocessors()]

    logger = Logger(agent.__class__.__name__, use_timestamp=True, seed=seed, results_dir=None)
    core = Core(agent, mdp, preprocessors=preprocessors)

    learn_params = dict(
        render=False,
        quiet=quiet
    )

    if n_steps is None and n_episodes is not None:
        learn_params['n_episodes'] = n_episodes
    elif n_steps is not None and n_episodes is None:
        learn_params['n_steps'] = n_steps
    else:
        raise AttributeError('Set parameter n_steps or n_episodes')

    eval_params = dict(
        render=False,
        quiet=quiet)

    if n_steps_test is None and n_episodes_test is not None:
        eval_params['n_episodes'] = n_episodes_test
    elif n_steps_test is not None and n_episodes_test is None:
        eval_params['n_steps'] = n_steps_test
    else:
        raise AttributeError('Set parameter n_steps_test or n_episodes_test')

    if not quiet:
        logger.strong_line()
        logger.info('Starting experiment with seed {}'.format(seed))
        logger.strong_line()

    if save_agent:
        best_agent = deepcopy(agent)
    J, R, V, E = compute_metrics(core, eval_params, agent_builder, env_builder)
    best_J, best_R, best_V, best_E = J, R, V, E
    epoch_J = [J]  # discounted reward
    epoch_R = [R]  # total reward
    epoch_V = [V]  # Value function
    epoch_E = [E]  # policy entropy
    
    if not quiet:
        print_metrics(logger, 0, J, R, V, E)

    for epoch in trange(n_epochs, disable=quiet, leave=False):
        try:
            core.learn(**learn_params, **agent_builder.get_fit_params())
        except:
            e = sys.exc_info()
            logger.error('EXECUTION FAILED: EPOCH {} SEED {}'.format(epoch, seed))
            logger.exception(e)
            sys.exit()

        J, R, V, E = compute_metrics(core, eval_params, agent_builder, env_builder)

        epoch_J.append(J)
        epoch_R.append(R)
        if agent_builder.compute_value_function:
            epoch_V.append(V)
        if agent_builder.compute_policy_entropy:
            epoch_E.append(E)

        # Save if best Agent
        if J > best_J:
            best_J = float(J)
            best_R = float(R)
            if agent_builder.compute_value_function:
                best_V = float(V)
            if agent_builder.compute_policy_entropy:
                best_E = float(E)
            if save_agent:
                best_agent = deepcopy(agent)

        if not quiet:
            print_metrics(logger, epoch+1, J, R, V, E)

    result = dict(
        J=np.array(epoch_J),
        R=np.array(epoch_R),
        score=[best_J, best_R, best_V])

    if save_agent:
        result['agent'] = best_agent

    if agent_builder.compute_value_function:
        result['V'] = np.array(epoch_V)
        result['score'].append(best_V)
    
    if agent_builder.compute_policy_entropy:
        result['E'] = np.array(epoch_E)
        result['score'].append(best_E)

    return result


def compute_metrics(core, eval_params, agent_builder, env_builder):
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

    if hasattr(J, 'item'):
        J = J.item()

    # Compute R
    R = np.mean(compute_J(dataset))

    if hasattr(R, 'item'):
        R = R.item()
    
    # Compute V
    V = None
    if agent_builder.compute_value_function:
        states = get_init_states(dataset)
        V = agent_builder.compute_Q(
            agent=core.agent,
            states=states)

        if hasattr(V, 'item'):
            V = V.item()
    
    # Compute Policy Entropy
    E = None
    if agent_builder.compute_policy_entropy:
        if agent_builder.compute_entropy_with_states:
            E = core.agent.policy.entropy(parse_dataset(dataset)[0])
        else:
            E = core.agent.policy.entropy()

    if hasattr(E, 'item'):
        E = E.item()
    
    return J, R, V, E


def print_metrics(logger, epoch, J, R, V, E):
    """
    Function that pretty prints the metrics on the standard output.

    Args:
        logger (Logger): MushroomRL logger object;
        epoch (int): the current epoch;
        J (float): the current value of J;
        R (float): the current value of R;
        V (float): the current value of V;
        E (float): the current value of E (Set None if not defined).

    """
    log_dict = dict(J=J, R=R)

    if E is not None:
        log_dict['E'] = E

    if V is not None:
        log_dict['V'] = V

    logger.epoch_info(epoch, **log_dict)
    logger.weak_line()
