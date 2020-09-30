import sys
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from mushroom_rl_benchmark.utils import be_range, get_init_states


def exec_run(agent_builder, env_builder, n_epochs, n_steps, n_steps_test=None, n_episodes_test=None, seed=None,
             quiet=True):
    if seed is not None:
        print('SEED', seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    cmp_E = agent_builder.compute_policy_entropy

    mdp = env_builder.build()
    if seed is not None:
        mdp.env.seed(seed)
    agent = agent_builder.build(mdp.info)
    preprocessors = [prepro(mdp.info) for prepro in agent_builder.get_preprocessors()]

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

    best_agent = agent
    J, R, Q, E = compute_metrics(core, eval_params, agent_builder, cmp_E)
    best_J, best_R, best_Q, best_E = J, R, Q, E
    epoch_Js = [J] # discounted reward
    epoch_Rs = [R] # total reward
    epoch_Qs = [Q] # Q Value
    epoch_Es = [E] # policy entropy
    
    if quiet is False:
        print_metrics(0, J, R, Q, E, start=True)

    for epoch in be_range(n_epochs, quiet):
        try:
            core.learn(n_steps=n_steps, n_steps_per_fit=agent_builder.get_n_steps_per_fit(), quiet=quiet)
        except:
            e = sys.exc_info()
            print('[ERROR] EXECUTION FAILED')
            print('EPOCH', epoch)
            print('SEED', seed)
            print(e)
            sys.exit()

        J, R, Q, E = compute_metrics(core, eval_params, agent_builder, cmp_E)

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

        if quiet is False:
            print_metrics(epoch, J, R, Q, E)

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


def compute_metrics(core, eval_params, agent_builder, cmp_E):
    dataset = core.evaluate(**eval_params)

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


def print_metrics(epoch, J, R, Q, E, start=False):
    if E is None:
        tqdm.write('{} OF EPOCH {}'.format('START' if start else 'END', str(epoch)))
        tqdm.write('J: {}, R: {}, Q: {}'.format(J, R, Q))
        tqdm.write('##################################################################################################')
    else:
        tqdm.write('{} OF EPOCH {}'.format('START' if start else 'END', str(epoch)))
        tqdm.write('J: {}, R: {}, Q: {}, E: {}'.format(J, R, Q, E))
        tqdm.write('##################################################################################################')
