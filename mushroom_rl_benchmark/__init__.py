__version__ = '1.0.0'

try:
    from .core.logger import BenchmarkLogger
    from .core.visualizer import BenchmarkVisualizer
    from .core.experiment_class import BenchmarkExperiment
    from .core.suite import BenchmarkSuite

    __all__ = [
        'BenchmarkLogger',
        'BenchmarkVisualizer',
        'BenchmarkExperiment',
        'BenchmarkSuite'
    ]
except ImportError as e:
    __all__ = []
