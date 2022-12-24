from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment_flat

from mushroom_rl_benchmark import BenchmarkExperiment


@single_experiment_flat
def experiment(agent: str = None,
               env: str = None,
               quiet: bool = True,
               save_agent: bool = False,
               seed: int = 0,
               results_dir: str = '/logs',
               **kwargs):

    exp = BenchmarkExperiment(agent, env, quiet=quiet, **kwargs)

    exp.run(save_agent, quiet, seed, results_dir)


if __name__ == '__main__':
    run_experiment(experiment)


