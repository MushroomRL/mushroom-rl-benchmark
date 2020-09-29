__version__ = '1.0.0'

try:
    from .logger import BenchmarkLogger
    from .visualizer import BenchmarkVisualizer
    from .experiment_class import BenchmarkExperiment
    from .suite import BenchmarkSuite

    __all__ = [
        'BenchmarkLogger',
        'BenchmarkVisualizer',
        'BenchmarkExperiment',
        'BenchmarkSuite'
    ]
except ImportError as e:
    __all__ = []
    