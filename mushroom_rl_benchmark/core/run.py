from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment_flat

from mushroom_rl_benchmark.core import BenchmarkExperiment


@single_experiment_flat
def experiment(agent: str = None,
               env: str = None,
               sweep_name: str = '',
               quiet: bool = True,
               show_progress_bar: bool = True,
               save_agent: bool = False,
               seed: int = 0,
               results_dir: str = '/logs',
               **kwargs):

    exp = BenchmarkExperiment(agent, env, results_dir,
                              sweep_name=sweep_name,
                              quiet=quiet,
                              show_progress_bar=show_progress_bar,
                              **kwargs)

    exp.run(save_agent, quiet, show_progress_bar, seed)


if __name__ == '__main__':
    run_experiment(experiment)


