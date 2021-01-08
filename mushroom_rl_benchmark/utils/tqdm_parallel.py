import joblib
from tqdm import tqdm


class TqdmParallel(joblib.Parallel):
    def __call__(self, *args, total=None, **kwargs):
        self._total = total
        with tqdm(total=total, leave=False) as self._progress_bar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._progress_bar.total = self.n_dispatched_tasks

        self._progress_bar.n = self.n_completed_tasks
        self._progress_bar.refresh()