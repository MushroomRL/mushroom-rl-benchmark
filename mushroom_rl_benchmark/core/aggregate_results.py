import numpy as np
from pathlib import Path

from samba.dcerpc.dcerpc import working

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
    work_dir = Path(res_dir, res_id)
    loader = BenchmarkDataLoader(work_dir)

    logger = Logger(work_dir.name, results_dir=work_dir.parent, log_console=log_console)
    logger.weak_line()
    logger.info(f'Environment: {work_dir.parent.name}, Agent: {res_id}')
    logger.info(f'path {work_dir}')

    has_entropy = (work_dir / 'E-0.npy').exists()
    logger.info(f'has entropy: {has_entropy}')

    has_value = (work_dir / 'V-0.npy').exists()
    logger.info(f'has value function: {has_value}')

    n_seeds = len([file for file in work_dir.glob(f"J-*.npy")])

    J_list = list()
    R_list = list()
    V_list = list()
    E_list = list()

    skip_cnt = 0

    failed_seeds = list()
    found_seeds = list()
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

            found_seeds.append(seed)

        except FileNotFoundError:
            failed_seeds.append(seed)
            skip_cnt += 1
        except Exception as e:
            logger.exception(e)

    if len(failed_seeds) > 0:
        logger.warning(f'NUMBER OF FAILED RUNS {len(failed_seeds)}/{n_seeds}')
        logger.warning(f'Failed seeds: {str(failed_seeds)}')

    J_len = np.array([len(J) for J in J_list])
    max_len = max(J_len)
    completed = np.argwhere(J_len == max_len).flatten()
    incomplete_seeds = np.argwhere(J_len < max_len).flatten()
    incomplete_count = len(incomplete_seeds)
    skip_cnt += len(J_list) - len(completed)

    J_np = np.array([J_list[i] for i in completed])
    R_np = np.array([R_list[i] for i in completed])
    if has_value:
        V_np = np.array([V_list[i] for i in completed])
    if has_entropy:
        E_np = np.array([E_list[i] for i in completed])

    if skip_cnt < n_seeds:
        if incomplete_count > 0:
            total = incomplete_count + len(completed)
            logger.warning(f'NUMBER OF INCOMPLETE RUNS: {incomplete_count}/{total}')
            logger.warning(f'Incomplete seeds: {str(np.array(found_seeds)[incomplete_seeds])}')
        else:
            logger.info('All runs succeeded')
        logger.log_numpy_array(J=J_np, R=R_np)

        if has_value:
            logger.log_numpy_array(V=V_np)
        if has_entropy:
            logger.log_numpy_array(E=E_np)

    else:
        logger.error('NO RUN SUCCEEDED')
