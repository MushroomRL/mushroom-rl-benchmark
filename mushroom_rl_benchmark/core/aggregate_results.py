import numpy as np
from pathlib import Path

from mushroom_rl.core import Logger

from mushroom_rl_benchmark.core import BenchmarkDataLoader


def aggregate_results(res_dir, res_id, log_console=False):
    """
    Function to aggregate the benchmark results from running in SLURM mode.
    Args:
        res_dir (str): path to the result directory;
        res_id (str): log id of the result directory;
        log_console (bool,False): whether to log the console output.

    """
    n_seeds = 2

    work_dir = Path(res_dir, res_id)
    loader = BenchmarkDataLoader(work_dir)

    logger = Logger(work_dir.name, results_dir=work_dir.parent, log_console=log_console)
    logger.strong_line()
    logger.info(f'Env: {work_dir.parent.name} Alg: {res_id}')
    logger.info(f'path {work_dir}')

    has_entropy = (work_dir / 'E-0.npy').exists()
    logger.info(f'has entropy: {has_entropy}')

    has_value = (work_dir / 'V-0.npy').exists()
    logger.info(f'has value function: {has_value}')

    J_list = list()
    R_list = list()
    V_list = list()
    E_list = list()

    skip_cnt = 0

    for seed in range(n_seeds):
        try:
            J = loader.load_run_file('J', seed)
            R = loader.load_run_file('R', seed)

            J_list.append(J)
            R_list.append(R)

            if has_value:
                V = loader.load_run_file('V', seed)
                V_list.append(V)
            if has_entropy:
                E = loader.load_run_file('E', seed)
                E_list.append(E)

            logger.info(f'Run {seed} OK')
        except Exception as e:
            logger.info(f'Run {seed} ERROR')
            skip_cnt += 1
            print(e)

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
            logger.warning(f'NUMBER OF FAILED RUNS: {skip_cnt}/{n_seeds}')
        else:
            logger.info('All runs succeeded')
        logger.log_numpy_array(J=J_np, R=R_np)

        if has_value:
            logger.log_numpy_array(V=V_np)
        if has_entropy:
            logger.log_numpy_array(E=E_np)

    else:
        logger.error('NO RUN SUCCEEDED')