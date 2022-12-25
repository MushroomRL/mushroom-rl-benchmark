import numpy as np
from pathlib import Path


class BenchmarkDataLoader:
    def __init__(self, path):
        self._path = Path(path)

    def load_run_file(self, name, seed):
        filename = f'{name}-{seed}.npy'

        return np.load(self._path / filename)

    def load_aggregate_file(self, name):
        filename = f'{name}.npy'

        return np.load(self._path / filename)

    def file_found(self, filename):
        file = self._path / f'{filename}.npy'

        return file.exists()

    @property
    def value_function_found(self):
        return self.file_found('V')

    @property
    def entropy_found(self):
        return self.file_found('E')
