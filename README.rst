********************
MushroomRL Benchmark
********************

.. image:: https://readthedocs.org/projects/mushroom-rl-benchmark/badge/?version=latest
    :target: https://mushroom-rl-benchmark.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**MushroomRL Benchmarking: Benchmarking tool for the MushroomRL library.**

.. contents:: **Contents of this document:**
   :depth: 2


What is MushroomRL Benchmark?
=============================

MushroomRL Benchmark is a benchmarking framework that aims to provide the RL research community with a powerful but easy
to use framework to design, execute and present scientifically sound experimentsfor deep RL algorithms. The benchmarking
framework builds on top of MushroomRL and utilizes the wide number of algorithms and environments that MushroomRL 
provides.

Installation
------------

You can do a minimal installation of ``mushroom_rl_benchmark`` with:

::

    $ pip install  -e .


Launch predefined benchmarks
============================

We provide a simple script `benchmark.py` to easily run benchmarks from configuration files.
You must have both mushroom-rl and mushroom-rl-benchmark packages installed.

The script for starting the benchmarks takes following arguments:

::

    usage: benchmark.py [-h] -e ENV [ENV ...] [-a ALGORITHM [ALGORITHM ...]] [-s SEEDS] [-x {sequential,parallel,slurm}] [-p PARAMETER_SWEEP] [-t] [-d]

    optional arguments:
      -h, --help            show this help message and exit

    benchmark parameters:
      -e ENV [ENV ...], --env ENV [ENV ...]
                            Environments to be used by the benchmark. Use 'all' to select all the available environments.
      -a ALGORITHM [ALGORITHM ...], --algorithm ALGORITHM [ALGORITHM ...]
                            Algorithms to be used by the benchmark. Use 'all' to select all the algorithms defined in the config file.
      -s SEEDS, --seeds SEEDS
                            Number of seed per experiment
      -x {sequential,parallel,slurm}, --execution_type {sequential,parallel,slurm}
                            Execution type for the benchmark.
      -p PARAMETER_SWEEP, --parameter-sweep PARAMETER_SWEEP
                            Sweep configuration file to be used by the benchmark.
      -t, --test            Flag to test the script and NOT execute the benchmark.
      -d, --demo            Flag to run a reduced version of the benchmark.





The agent and environment parameters used for benchmarking the agents on an environment are located in

::

    cfg/env/*

The available parameter sweeps can be found in:

::

    cfg/sweep/*

The parameters used to configure the main folder, the log id and the execution backend (parallel or slurm) and are
located in:

::

    cfg/suite.yaml

The parameters to customize the plots are located in:

::

    cfg/plots.yaml


Launch benchmarks
-----------------

To run a reduced benchmark locally call the script like this:

.. code:: shell

    $ ./benchmark.py -e <EnvironmentName> -d

To run a reduced benchmark on a SLURM cluster call the script like this:

.. code:: shell

    $ ./benchmark.py -e <EnvironmentName> -x slurm -d

To run the full benchmark for all environments, on a SLURM cluster call the script like this:

.. code:: shell

    $ ./benchmark.py -e all -x slurm

Create Plots
------------

If you need to create the plots for a benchmarking folder, you can call the following script

.. code:: shell

    $ ./create_plots -d <BenchmarkDir>

where `BenchmarkDir` is the directory of your benchmark, e.g. "logs/benchmark"
