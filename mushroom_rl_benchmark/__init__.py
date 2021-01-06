__version__ = '1.0.0'

try:
    from .core import BenchmarkLogger, BenchmarkVisualizer, BenchmarkSuiteVisualizer,\
        BenchmarkExperiment, BenchmarkSuite

    __all__ = [
        'BenchmarkLogger',
        'BenchmarkVisualizer',
        'BenchmarkSuiteVisualizer',
        'BenchmarkExperiment',
        'BenchmarkSuite'
    ]
except ImportError as e:
    __all__ = []
