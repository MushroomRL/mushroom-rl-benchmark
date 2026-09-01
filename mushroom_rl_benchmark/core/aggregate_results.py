import numpy as np
from pathlib import Path

from mushroom_rl.core import Logger

from mushroom_rl_benchmark.core import BenchmarkDataLoader


def aggregate_results(res_dir, log_console=False):
    """
    Function to aggregate the benchmark results.

    Args:
        res_dir (str): path to the result directory;
        log_console (bool, False): whether to log the console output.

    Returns:
        A dictionary containing the aggregated metric arrays.

    """
    work_dir = Path(res_dir)
    loader = BenchmarkDataLoader(work_dir)

    logger_name = f'{work_dir.parent.name}/{work_dir.name}'
    logger = Logger(logger_name, results_dir=work_dir.parent.parent,
                    log_console=log_console, log_file_name=work_dir.name)
    logger.weak_line()
    logger.info(f'Environment: {work_dir.parent.name}, Agent: {work_dir.name}')
    logger.info(f'path {work_dir}')

    has_entropy = any(work_dir.glob('E-*.npy'))
    logger.info(f'has entropy: {has_entropy}')

    has_value = any(work_dir.glob('V-*.npy'))
    logger.info(f'has value function: {has_value}')

    prefix = 'J-'
    seeds = sorted(int(path.stem[len(prefix):]) for path in work_dir.glob('J-*.npy')
                   if path.stem[len(prefix):].isdigit())

    J_list = list()
    R_list = list()
    V_list = list()
    E_list = list()

    failed_seeds = list()
    found_seeds = list()
    for seed in seeds:
        try:
            J = loader.load_run_file('J', seed)
            R = loader.load_run_file('R', seed)

            J_list.append(J)
            R_list.append(R)
            if has_value:
                V_list.append(loader.load_run_file('V', seed))
            if has_entropy:
                E_list.append(loader.load_run_file('E', seed))

            found_seeds.append(seed)

        except FileNotFoundError:
            failed_seeds.append(seed)

    if failed_seeds:
        logger.warning(f'NUMBER OF FAILED RUNS {len(failed_seeds)}/{len(seeds)}')
        logger.warning(f'Failed seeds: {str(failed_seeds)}')

    if not J_list:
        raise RuntimeError(f'No complete runs found in {work_dir}')

    J_len = np.array([len(J) for J in J_list])
    max_len = max(J_len)
    completed = np.argwhere(J_len == max_len).flatten()
    incomplete_seeds = np.argwhere(J_len < max_len).flatten()

    J_np = np.array([J_list[i] for i in completed])
    R_np = np.array([R_list[i] for i in completed])
    results = dict(J=J_np, R=R_np)
    if has_value:
        results['V'] = np.array([V_list[i] for i in completed])
    if has_entropy:
        results['E'] = np.array([E_list[i] for i in completed])

    if len(incomplete_seeds) > 0:
        logger.warning(f'NUMBER OF INCOMPLETE RUNS: {len(incomplete_seeds)}/{len(J_list)}')
        logger.warning(f'Incomplete seeds: {str(np.array(found_seeds)[incomplete_seeds])}')
    else:
        logger.info('All runs succeeded')
    logger.log_numpy_array(**results)

    return results
