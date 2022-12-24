__version__ = '1.1.0'

try:
    from .core import BenchmarkConfiguration, BenchmarkExperiment, BenchmarkSuite

except ImportError as e:
    pass
