import os
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pkgutil
import subprocess

import numpy as np
from tqdm import tqdm, trange
import multiprocessing
from joblib import Parallel, delayed

from mushroom_rl_benchmark.utils import extract_arguments
from mushroom_rl_benchmark.experiment import exec_run
from mushroom_rl_benchmark.experiment.slurm import create_slurm_script, generate_slurm, make_arguments
from mushroom_rl_benchmark.core.visualizer import BenchmarkVisualizer


class BenchmarkExperiment:
    """
    Class to create and run an experiment using MushroomRL

    """
    def __init__(self, agent_builder, env_builder, logger):
        """
        Constructor.

        Args:
            agent_builder (AgentBuilder): instance of a specific agent builder
            env_builder (EnvironmentBuilder): instance of an environment builder
            logger (BenchmarkLogger): instance of a benchmark logger

        Returns:
            benchmark_experiment: instance of a benchmark experiment

        """
        self.agent_builder = agent_builder
        self.env_builder = env_builder
        self.logger = logger
        self.start_time = 0
        self.stop_time = 0

        self.Js = list()
        self.Qs = list()
        self.Rs = list()
        self.policy_entropies = list()
        self.config = dict()
        self.stats = dict(best_J=float("-inf"))

    def run(self, exec_type='sequential', **run_params):
        """
        Execute the experiment.

        Args:
            exec_type (str, 'sequential'): type of executing the experiment [sequential|parallel|slurm]
            **run_params: parameters for the selected execution type

        """
        try:
            run_fn = getattr(self, 'run_{}'.format(exec_type))
        except AttributeError as e: 
            self.logger.exception(e)
            raise ValueError("exec_type must be 'sequential', 'parallel' or 'slurm'")
        self.logger.info('Running BenchmarkExperiment {}'.format(exec_type))
        run_fn(**run_params)

    def run_sequential(self, n_runs, n_runs_completed=0, save_plot=True, **run_params):
        """
        Execute the experiment sequential.

        Args:
            n_runs (int): number of total runs of the experiment
            n_runs_completed (int): number of completed runs of the experiment
            save_plot (bool): select if a plot of the experiment should be saved to the log directory
            **run_params: parameters for executing a benchmark run

        """
        self.start_timer()
        
        self.save_builders()

        self.set_and_save_config(
            agent_type=self.agent_builder.__class__.__name__,
            n_runs=n_runs,
            n_runs_completed=n_runs_completed,
            run_parallel=False,
            use_threading=False,
            **run_params)
        
        cmp_E = self.agent_builder.compute_policy_entropy

        for run in trange(n_runs_completed, n_runs):
            result = exec_run(self.agent_builder, self.env_builder, quiet=False, **run_params)
            self.extend_and_save_Js([result['Js']])
            self.extend_and_save_Rs([result['Rs']])
            self.extend_and_save_Qs([result['Qs']])
            if cmp_E:
                self.extend_and_save_policy_entropies([result['Es']])
            new_score = result['score']
            new_agent = result['builders']
            if new_score[0] > self.stats['best_J']:
                self.set_and_save_stats(
                    best_J=new_score[0],
                    best_R=new_score[1],
                    best_Q=new_score[2])
                if cmp_E:
                    self.set_and_save_stats(best_E=new_score[3])
                self.logger.save_best_agent(new_agent)
            self.set_and_save_config(n_runs_completed=(run+1))
        self.stop_timer()

        if save_plot:
            self.save_plot()

    def run_parallel(self, n_runs, n_runs_completed=0, threading=False,
                     save_plot=True, max_concurrent_runs=None, **run_params):
        """
        Execute the experiment in parallel threads.

        Args:
            n_runs (int): number of total runs of the experiment
            n_runs_completed (int): number of completed runs of the experiment
            threading (bool): select to use threads instead of processes
            save_plot (bool): select if a plot of the experiment should be saved to the log directory
            max_concurrent_runs (int): maximum number of concurrent runs (Default: number of cores)
            **run_params: parameters for executing a benchmark run

        """
        self.start_timer()
        self.save_builders()
        
        if max_concurrent_runs is None:
            n_cpu = multiprocessing.cpu_count()
            max_concurrent_runs = n_runs if n_runs < n_cpu else n_cpu

        self.logger.info('Number of used cores: {}'.format(max_concurrent_runs))

        parallel_settings = dict()
        parallel_settings['n_jobs'] = max_concurrent_runs
        if threading:
            parallel_settings['prefer'] = 'threads'

        self.set_and_save_config(
            agent_type=self.agent_builder.__class__.__name__,
            n_runs_completed=n_runs_completed,
            n_runs=n_runs,
            max_concurrent_runs=max_concurrent_runs,
            use_threading=threading,
            run_parallel=True,
            **run_params
        )
        
        cmp_E = self.agent_builder.compute_policy_entropy

        self.logger.info('Starting experiment ...')

        t = tqdm(total=n_runs)
        with Parallel(**parallel_settings) as parallel:
            remaining = n_runs
            n_parallel = max_concurrent_runs
            while remaining > 0:
                if remaining < max_concurrent_runs:
                    n_parallel = remaining
                runs = parallel(
                    delayed(exec_run)(self.agent_builder.copy(), self.env_builder.copy(),
                                      seed=seed.item(), quiet=True, **run_params)
                    for seed in np.random.randint(100000000, size=n_parallel)
                )
                
                run_Js = list()
                run_Rs = list()
                run_Qs = list()
                run_Es = list()
                new_score = [float("-inf"), 0, 0, 0] # J, R, Q, E
                new_agent = None

                for run in runs:
                    # Collect J, R, Q and E
                    run_Js.append(run['Js'])
                    run_Rs.append(run['Rs'])
                    run_Qs.append(run['Qs'])
                    if cmp_E:
                        run_Es.append(run['Es'])
                    
                    # Check for best Agent (depends on J)
                    if run['score'][0] > new_score[0]:
                        new_score = run['score']
                        new_agent = run['agent']

                self.extend_and_save_Js(run_Js)
                self.extend_and_save_Rs(run_Rs)
                self.extend_and_save_Qs(run_Qs)
                if cmp_E:
                    self.extend_and_save_policy_entropies(run_Es)

                if new_score[0] > self.stats['best_J']:
                    self.set_and_save_stats(
                        best_J=new_score[0],
                        best_R=new_score[1],
                        best_Q=new_score[2])
                    if cmp_E:
                        self.set_and_save_stats(best_E=new_score[3])
                    self.logger.save_best_agent(new_agent)

                t.update(n_parallel)
                remaining -= n_parallel
                self.set_and_save_config(n_runs_completed=(n_runs-remaining))
        self.stop_timer()

        self.logger.info('Finished experiment.')

        if save_plot:
            self.save_plot()

    def run_slurm(self, n_runs, n_runs_completed=0, aggregation_job=True, aggregate_hours=3,
                  aggregate_minutes=0, aggregate_seconds=0, demo=False, **run_params):
        """
        Execute the experiment with SLURM.

        Args:
            n_runs (int): number of total runs of the experiment
            n_runs_completed (int): number of completed runs of the experiment
            aggregation_job (bool): select if an aggregation job should be scheduled
            aggregate_hours (int): maximum number of hours for the aggregation job (Default: 3)
            aggregate_minutes (int): maximum number of minutes for the aggregation job (Default: 0)
            aggregate_seconds (int): maximum number of seconds for the aggregation job (Default: 0)
            demo (bool): select if the experiment should be run in demo mode
            **run_params: parameters for executing a benchmark run

        """

        exec_params = extract_arguments(run_params, exec_run)
        slurm_params = extract_arguments(run_params, create_slurm_script)
        slurm_params.update(extract_arguments(run_params, generate_slurm))

        # Create SLURM Script for experiment runs
        log_dir = os.path.abspath(self.logger.get_path())
        log_id = self.logger.get_log_id()
        python_file = pkgutil.get_loader("mushroom_rl_benchmark.experiment.slurm.run_script").path
        script_path = create_slurm_script(
            slurm_path=log_dir, 
            slurm_script_name='slurm_run.sh',
            exp_name=log_id, 
            exp_dir_slurm=log_dir,
            n_exp=n_runs,
            python_file=python_file,
            **slurm_params)

        # Create SLURM Script for experiment aggregation
        python_file_aggregate = pkgutil.get_loader("mushroom_rl_benchmark.experiment.slurm.aggregate_results").path
        script_path_aggregate = create_slurm_script(
            slurm_path=log_dir,
            slurm_script_name='slurm_aggregate.sh',
            exp_name='{}_aggregate'.format(log_id), 
            exp_dir_slurm=log_dir,
            python_file=str(python_file_aggregate),
            hours=aggregate_hours,
            minutes=aggregate_minutes,
            seconds=aggregate_seconds)

        # save builder and aggregate script to path
        self.save_builders()

        self.set_and_save_config(
            agent_type=self.agent_builder.__class__.__name__,
            n_runs_completed=n_runs_completed,
            n_runs=n_runs,
            max_concurrent_runs=None,
            use_threading=False,
            run_parallel=False,
            run_slurm=True,
            **exec_params
        )

        # submit job array with n_exp runs
        command_line_arguments = make_arguments(
            log_dir=log_dir,
            **exec_params
        )

        command_line_arguments_aggregate = make_arguments(
            log_dir=self.logger.get_log_dir() ,
            log_id=log_id
        )
        
        command = "sbatch --parsable {} {}".format(script_path, command_line_arguments)

        if demo:
            self.logger.info(command)
        else:
            slurm_job_id = subprocess.getoutput(command).split(' ')[-1]
            self.logger.info('slurm_job_id: ' + slurm_job_id)
            if aggregation_job:
                command_aggregate = "sbatch --parsable --dependency=afterok:{} {} {}".format(slurm_job_id,
                                                                                             script_path_aggregate,
                                                                                             command_line_arguments_aggregate)
                slurm_job_id_aggregate = subprocess.getoutput(command_aggregate).split(' ')[-1]
                self.logger.info('slurm_job_id (aggregate): ' + slurm_job_id_aggregate)
            else:
                self.logger.info('No aggregation job scheduled.')
    
    def reset(self):
        """
        Reset the internal state of the experiment.

        """
        self.Js = list()
        self.Qs = list()
        self.Rs = list()
        self.policy_entropies = list()

    def resume(self, logger):
        """
        Resume an experiment from disk

        """
        raise NotImplementedError('This method was not yet implemented.')

    def start_timer(self):
        """
        Start the timer.

        """
        self.start_time = time.time()

    def stop_timer(self):
        """
        Stop the timer.

        """
        self.stop_time = time.time()
        self.set_and_save_stats(
            execution_time_sec=(self.stop_time - self.start_time)
        )

    def save_builders(self):
        """
        Save agent and environment builder to the log directory.

        """
        self.logger.save_agent_builder(self.agent_builder)
        self.logger.save_environment_builder(self.env_builder)

    def extend_and_save_Js(self, Js):
        """
        Extend Js with another datapoint and save the current state to the log directory.

        """
        self.Js.extend(Js)
        self.logger.save_Js(self.Js)

    def extend_and_save_Rs(self, Rs):
        """
        Extend Rs with another datapoint and save the current state to the log directory.

        """
        self.Rs.extend(Rs)
        self.logger.save_Rs(self.Rs)

    def extend_and_save_Qs(self, Qs):
        """
        Extend Qs with another datapoint and save the current state to the log directory.

        """
        self.Qs.extend(Qs)
        self.logger.save_Qs(self.Qs)

    def extend_and_save_policy_entropies(self, policy_entropies):
        """
        Extend Es with another datapoint and save the current state to the log directory.

        """
        self.policy_entropies.extend(policy_entropies)
        self.logger.save_policy_entropies(self.policy_entropies)

    def set_and_save_config(self, **settings):
        """
        Save the experiment configuration to the log directory.

        """
        self.config.update(settings)
        self.logger.save_config(self.config)

    def set_and_save_stats(self, **info):
        """
        Save the run statistics to the log directory.

        """
        self.stats.update(info)
        self.logger.save_stats(self.stats)

    def save_plot(self):
        """
        Save the result plot to the log directory.

        """
        visualizer = BenchmarkVisualizer(self.logger)
        visualizer.save_report()
    
    def show_plot(self):
        """
        Display the result plot.

        """
        visualizer = BenchmarkVisualizer(self.logger)
        visualizer.show_report()

