#!/usr/bin/env python3

from pathlib import Path

from mushroom_rl.core import Logger
from mushroom_rl_benchmark import BenchmarkLogger, BenchmarkVisualizer
from mushroom_rl_benchmark.experiment.slurm import read_arguments_aggregate


def aggregate_results(res_dir, res_id, console_log_dir=None):
    """
    Function to aggregate the benchmark results from running in SLURM mode.

    Args:
        res_dir (str): path to the result directory;
        res_id (str): log id of the result directory;
        console_log_dir (str,None): path to be used to log console.
    
    """
    work_dir = Path(res_dir, res_id)
    aggregation_name = work_dir.name + '_' + work_dir.parent.name
    console = Logger(aggregation_name, results_dir=console_log_dir,
                     log_console=console_log_dir is not None)
    console.strong_line()
    console.info(aggregation_name)

    # check if results are aggregated
    
    dir_name = 'run'

    run_dirs = list(work_dir.glob('{}_*'.format(dir_name)))

    has_entropy = (work_dir / '{}_0/entropy.pkl'.format(dir_name)).exists()
    console.info(f'has entropy: {has_entropy}')

    J = list()
    R = list()
    V = list()
    E = list()
    best_J = float("-inf")
    best_stats = None
    best_agent = None

    skip_cnt = 0

    for run_dir in run_dirs: 
        logger = BenchmarkLogger(log_dir=str(run_dir.parent), log_id=str(run_dir.name), use_timestamp=False)

        try:
            J.extend(logger.load_J())
            R.extend(logger.load_R())
            V.extend(logger.load_V())
            if has_entropy:
                E.extend(logger.load_entropy())
            # stats = logger.load_stats()
            # if stats['best_J'] > best_J:
            #     best_stats = stats
            #     if logger.exists_best_agent():
            #         best_agent = logger.load_best_agent()
            console.info(run_dir.name + " OK")
        except Exception as e:
            console.error(run_dir.name + " ERROR")
            #console.exception(e)
            skip_cnt += 1

    if skip_cnt < len(run_dirs):
        if skip_cnt > 0:
            console.warning(f'NUMBER OF FAILED RUNS: {skip_cnt}/{len(run_dirs)}')

        console.info('Env: ' + work_dir.name + ' alg: ' + res_id)

        logger = BenchmarkLogger(log_dir=res_dir, log_id=res_id, use_timestamp=False)

        logger.save_J(J)
        logger.save_R(R)
        logger.save_V(V)
        if has_entropy:
            logger.save_entropy(E)
        # logger.save_stats(best_stats)
        if best_agent is not None:
            logger.save_best_agent(best_agent)

        visualizer = BenchmarkVisualizer(logger)
        visualizer.save_report()
    else:
        console.error('NO RUN SUCCEEDED')


if __name__ == '__main__':
    res_dir, res_id = read_arguments_aggregate()
    aggregate_results(res_dir, res_id)
