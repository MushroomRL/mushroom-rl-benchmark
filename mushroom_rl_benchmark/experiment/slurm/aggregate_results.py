#!/usr/bin/env python3

from pathlib import Path

from mushroom_rl_benchmark import BenchmarkLogger, BenchmarkVisualizer
from mushroom_rl_benchmark.experiment.slurm import read_arguments_aggregate


def run(res_dir, res_id):
    """
    Function to aggregate the benchmark results from running in SLURM mode.

    Args:
        res_dir (str): path to the result directory;
        res_id (str): log id of the result directory.
    
    """
    work_dir = Path(res_dir, res_id)

    # check if results are aggregated
    
    dir_name = 'run'

    run_dirs = list(work_dir.glob('{}_*'.format(dir_name)))
    print(run_dirs)

    has_entropy = (work_dir / '{}_0/policy_entropies.pkl'.format(dir_name)).exists()
    print('has entropy:', has_entropy)

    Js = list()
    Rs = list()
    Qs = list()
    Es = list()
    best_J = float("-inf")
    best_stats = None
    best_agent = None

    skip_cnt = 0

    for run_dir in run_dirs: 
        logger = BenchmarkLogger(log_dir=str(run_dir.parent), log_id=str(run_dir.name), use_timestamp=False)
        print(run_dir, end=' ')
        if not logger.exists_best_agent(): 
            print("ABORT")
            skip_cnt += 1
            continue
        else:
            print("EXISTS")
        Js.extend(logger.load_J())
        Rs.extend(logger.load_R())
        Qs.extend(logger.load_V())
        if has_entropy:
            Es.extend(logger.load_entropy())
        stats = logger.load_stats()
        if stats['best_J'] > best_J:
            best_stats = stats
            best_agent = logger.load_best_agent()
    
    if skip_cnt > 0: print('NUMBER OF FAILED RUNS:', '{}/{}'.format(skip_cnt, len(run_dirs)))
    
    print('Name:', res_id)
    print('Directory:', res_dir)

    logger = BenchmarkLogger(log_dir=res_dir, log_id=res_id, use_timestamp=False)

    logger.save_J(Js)
    logger.save_R(Rs)
    logger.save_V(Qs)
    if has_entropy:
        logger.save_entropy(Es)
    logger.save_stats(best_stats)
    logger.save_best_agent(best_agent)

    visualizer = BenchmarkVisualizer(logger)
    visualizer.save_report()


if __name__ == '__main__':
    res_dir, res_id = read_arguments_aggregate()
    run(res_dir, res_id)
