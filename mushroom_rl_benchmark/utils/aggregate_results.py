import numpy as np
from pathlib import Path

from mushroom_rl.core import Logger


def _load_file(path, name, seed):
    filename = f'{name}-{seed}.npy'

    return np.load(path / filename)


def _save_file(path, name, value):
    filename = f'{name}.npy'

    return np.save(path / filename, value)


def aggregate_results(res_dir, res_id, console_log_dir=None):
    """
    Function to aggregate the benchmark results from running in SLURM mode.
    Args:
        res_dir (str): path to the result directory;
        res_id (str): log id of the result directory;
        console_log_dir (str,None): path to be used to log console.

    """
    n_seeds = 2

    work_dir = Path(res_dir, res_id)
    aggregation_name = work_dir.name + '_' + work_dir.parent.name
    console = Logger(aggregation_name, results_dir=console_log_dir,
                     log_console=console_log_dir is not None)
    console.strong_line()
    console.info(f'Env: {work_dir.parent.name} Alg: {res_id}')
    console.info(f'path {work_dir}')

    has_entropy = (work_dir / 'E-0.npy').exists()
    console.info(f'has entropy: {has_entropy}')

    has_value = (work_dir / 'V-0.npy').exists()
    console.info(f'has value function: {has_value}')

    J_list = list()
    R_list = list()
    V_list = list()
    E_list = list()

    skip_cnt = 0

    for seed in range(n_seeds):
        try:
            J = _load_file(work_dir, 'J', seed)
            R = _load_file(work_dir, 'R', seed)

            J_list.append(J)
            R_list.append(R)

            if has_value:
                V = _load_file(work_dir, 'V', seed)
                V_list.append(V)
            if has_entropy:
                E = _load_file(work_dir, 'E', seed)
                E_list.extend(E)

            console.info(f'Run {seed} OK')
        except Exception:
            console.info(f'Run {seed} ERROR')
            skip_cnt += 1

    J_len = np.array([len(J) for J in J_list])
    max_len = max(J_len)
    completed = np.argwhere(J_len == max_len).flatten()
    skip_cnt += len(J_list) - len(completed)

    J_np = np.array([J_list[i] for i in completed])
    R_np = np.array([R_list[i] for i in completed])
    if has_value:
        V_np = np.array([V_list[i] for i in completed])
    if has_entropy:
        E_np = np.array([E_list[i] for i in completed])

    if skip_cnt < n_seeds:
        if skip_cnt > 0:
            console.warning(f'NUMBER OF FAILED RUNS: {skip_cnt}/{n_seeds}')

        _save_file(work_dir, 'J', J_np)
        _save_file(work_dir, 'R', R_np)

        if has_value:
            _save_file(work_dir, 'V', V_np)
        if has_entropy:
            _save_file(work_dir, 'E', E_np)
    else:
        console.error('NO RUN SUCCEEDED')