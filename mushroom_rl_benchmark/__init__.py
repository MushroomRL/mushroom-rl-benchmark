__version__ = '1.0.0'

try:
    from .core import BenchmarkLogger, BenchmarkVisualizer, BenchmarkExperiment, BenchmarkSuite

    __all__ = [
        'BenchmarkLogger',
        'BenchmarkVisualizer',
        'BenchmarkExperiment',
        'BenchmarkSuite'
    ]
except ImportError as e:
    __all__ = []
