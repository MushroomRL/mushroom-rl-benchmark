import matplotlib.pyplot as plt
import numpy as np

from mushroom_rl_benchmark.core import BenchmarkSuiteVisualizer


def test_plot_settings_are_selected_by_environment_and_metric(tmp_path):
    experiment_dir = tmp_path / 'Pendulum-v1' / 'PPO'
    experiment_dir.mkdir(parents=True)
    np.save(experiment_dir / 'J.npy', [[-10.0, -5.0], [-8.0, -4.0]])

    settings = {'Pendulum-v1': {'J': {'y_limits': {'ymin': -20.0, 'ymax': 0.0},
                                      'legend': {'ncol': 1}}}}
    visualizer = BenchmarkSuiteVisualizer(tmp_path, settings=settings)

    figure = visualizer.get_report('Pendulum-v1', 'J')

    assert figure.axes[0].get_ylim() == (-20.0, 0.0)
    plt.close(figure)
