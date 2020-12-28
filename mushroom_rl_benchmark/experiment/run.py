import sys
import torch
import numpy as np
from copy import deepcopy

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core.logger import Logger

from mushroom_rl_benchmark.utils import be_range, get_init_states


def exec_run(agent_builder, env_builder, n_epochs, n_steps, n_steps_test=None, n_episodes_test=None, seed=None,
             quiet=True, **kwargs):
    """
    Function that handles the execution of an experiment run.

    Args:
        agent_builder (AgentBuilder): agent builder to spawn an agent;
        env_builder (EnvironmentBuilder): environment builder to spawn an environment;
        n_epochs (int): number of epochs;
        n_steps (int): number of steps per epoch;
        n_steps_test (int, None): number of steps for testing;
        n_episodes_test (int, None): number of episodes for testing;
        seed (int, None): the seed;
        quiet (bool, True): select if run should print execution information.

    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    cmp_E = agent_builder.compute_policy_entropy

    mdp = env_builder.build()
    if hasattr(mdp, 'env'):
        mdp.env.seed(seed)
    agent = agent_builder.build(mdp.info)
    preprocessors = [prepro(mdp.info) for prepro in agent_builder.get_preprocessors()]

    logger = Logger(agent.__class__.__name__, use_timestamp=True, seed=seed, results_dir=None)
    core = Core(agent, mdp, preprocessors=preprocessors)

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

    best_agent = agent
    J, R, Q, E = compute_metrics(core, eval_params, agent_builder, env_builder,
                                 cmp_E)
    best_J, best_R, best_Q, best_E = J, R, Q, E
    epoch_Js = [J] # discounted reward
    epoch_Rs = [R] # total reward
    epoch_Qs = [Q] # Q Value
    epoch_Es = [E] # policy entropy
    
    if not quiet:
        print_metrics(logger, 0, J, R, Q, E)

    for epoch in be_range(n_epochs, quiet):
        try:
            core.learn(n_steps=n_steps, n_steps_per_fit=agent_builder.get_n_steps_per_fit(), quiet=quiet)
        except:
            e = sys.exc_info()
            logger.error('EXECUTION FAILED: EPOCH {} SEED {}'.format(epoch, seed))
            logger.exception(e)
            sys.exit()

        J, R, Q, E = compute_metrics(core, eval_params, agent_builder,
                                     env_builder, cmp_E)

        epoch_Js.append(J)
        epoch_Rs.append(R)
        epoch_Qs.append(Q)
        if cmp_E:
            epoch_Es.append(E)

        # Save if best Agent
        if J > best_J:
            best_J = float(J)
            best_R = float(R)
            best_Q = float(Q)
            if cmp_E:
                best_E = float(E)
            best_agent = deepcopy(agent)

        if not quiet:
            print_metrics(logger, epoch+1, J, R, Q, E)

    result = dict(
        Js=np.array(epoch_Js),
        Qs=np.array(epoch_Qs),
        Rs=np.array(epoch_Rs),
        agent=best_agent.copy(),
        score=[best_J, best_R, best_Q])
    
    if cmp_E:
        result['Es'] = np.array(epoch_Es)
        result['score'].append(best_E)

    return result


def compute_metrics(core, eval_params, agent_builder, env_builder, cmp_E):
    """
    Function to compute the metrics.

    Args:
        eval_params (dict): parameters for running the evaluation;
        agent_builder (AgentBuilder): the agent builder;
        env_builder (EnvironmentBuilder): environment builder to spawn an environment;
        cmp_E (bool): select if policy entropy should be computed.

    """

    agent_builder.set_eval_mode(core.agent, True)
    env_builder.set_eval_mode(core.mdp, True)
    dataset = core.evaluate(**eval_params)
    agent_builder.set_eval_mode(core.agent, False)
    env_builder.set_eval_mode(core.mdp, False)

    # Compute J
    J = np.mean(compute_J(dataset, core.mdp.info.gamma))

    # Compute R
    R = np.mean(compute_J(dataset))
    
    # Compute Q
    states = get_init_states(dataset)
    Q = agent_builder.compute_Q(
        agent=core.agent, 
        states=states)
    
    # Compute Policy Entropy
    E = None
    if cmp_E:
        if agent_builder.compute_entropy_with_states:
            E = core.agent.policy.entropy(parse_dataset(dataset)[0])
        else:
            E = core.agent.policy.entropy()
    
    return J, R, Q, E


def print_metrics(logger, epoch, J, R, Q, E):
    """
    Function that pretty prints the metrics on the standard output.

    Args:
        logger (Logger): MushroomRL logger object;
        epoch (int): the current epoch;
        J (float): the current value of J;
        R (float): the current value of R;
        Q (float): the current value of Q;
        E (float): the current value of E (Set None if not defined).

    """
    if E is None:
        logger.epoch_info(epoch, J=J, R=R, Q=Q)
    else:
        logger.epoch_info(epoch, J=J, R=R, Q=Q, E=E)
    logger.weak_line()
