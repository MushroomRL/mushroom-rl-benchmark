import os

from mushroom_rl_benchmark import BenchmarkLogger
from mushroom_rl_benchmark.experiment import exec_run
from mushroom_rl_benchmark.experiment.slurm import read_arguments_run

if __name__ == '__main__':

    log_dir, run_args = read_arguments_run()

    log_id = 'run_' + str(run_args['seed'])

    agent_builder = BenchmarkLogger._load_pickle(os.path.join(log_dir, 'agent_builder.pkl'))
    env_builder = BenchmarkLogger._load_pickle(os.path.join(log_dir, 'environment_builder.pkl'))

    logger = BenchmarkLogger(
        log_dir=log_dir,
        log_id=log_id,
        use_timestamp=False
    )

    logger.info('Starting experiment.')

    result = exec_run(agent_builder, env_builder, **run_args)

    logger.info('Saving result.')

    cmp_E = agent_builder.compute_policy_entropy

    logger.save_J([result['J']])
    logger.save_R([result['R']])
    logger.save_V([result['V']])
    if cmp_E:
        logger.save_entropy([result['E']])
    new_score = result['score']

    stats = dict(
        best_J=new_score[0],
        best_R=new_score[1],
        best_Q=new_score[2])

    if cmp_E:
        stats.update(dict(best_E=new_score[3]))
    logger.save_stats(stats=stats)

    if 'agent' in result:
        new_agent = result['agent']
        logger.save_best_agent(new_agent)

    logger.info('Finished execution.')
