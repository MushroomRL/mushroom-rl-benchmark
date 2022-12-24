__version__ = '2.0.0'

try:
    from .core import BenchmarkConfiguration, BenchmarkExperiment, BenchmarkSuite

except ImportError as e:
    pass
