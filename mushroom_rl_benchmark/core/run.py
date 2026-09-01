from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mushroom_rl_benchmark.core import BenchmarkExperiment


@hydra.main(version_base=None, config_path=None, config_name='benchmark')
def experiment(cfg: DictConfig):
    experiment_params = OmegaConf.to_container(cfg.experiment, resolve=True)
    run_params = experiment_params.pop('run_params')
    run_params.pop('n_runs', None)
    env_id = experiment_params.pop('env_id')

    exp = BenchmarkExperiment(results_dir=Path(cfg.output_root) / env_id,
                              log_console=bool(cfg.log_console), **experiment_params, **run_params)
    exp.run(seed=int(cfg.seed))


if __name__ == '__main__':
    experiment()
