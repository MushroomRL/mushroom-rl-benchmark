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

MushroomRL Benchmark is a benchmarking framework that aims to provide the RL research community with a powerful but
easy-to-use framework to design, execute and present scientifically sound experiments for deep RL algorithms. The
benchmarking framework builds on MushroomRL and utilizes the wide range of algorithms and environments that MushroomRL
provides.

Installation
------------

Install ``mushroom_rl_benchmark`` and the supported MushroomRL environments with:

::

    $ pip install -e .


Documentation dependencies follow MushroomRL's documentation setup:

::

    $ pip install -r docs/requirements.txt


Launch predefined benchmarks
============================

We provide a simple script `benchmark.py` to easily run benchmarks from configuration files.
You must have both mushroom-rl and mushroom-rl-benchmark packages installed.

The script for starting the benchmarks takes the following arguments:

::

    usage: benchmark.py [-h] -e ENV [ENV ...] [-a ALGORITHM [ALGORITHM ...]] [-s SEEDS] [-x {sequential,parallel,slurm}] [-t] [-d] [-o OUTPUT_DIR] [--quiet] [--override HYDRA_OVERRIDE]

    optional arguments:
      -h, --help            show this help message and exit

    benchmark parameters:
      -e ENV [ENV ...], --env ENV [ENV ...]
                            Environments to be used by the benchmark. Use 'all' to select all the available environments.
      -a ALGORITHM [ALGORITHM ...], --algorithm ALGORITHM [ALGORITHM ...]
                            Algorithms to be used by the benchmark. Use 'all' to select all the algorithms defined in the config file.
      -s SEEDS, --seeds SEEDS
                            Number of seeds per experiment
      -x {sequential,parallel,slurm}, --execution-type {sequential,parallel,slurm}
                            Execution type for the benchmark.
      -t, --test            Flag to test the script and NOT execute the benchmark.
      -d, --demo            Flag to run a reduced version of the benchmark.
      -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                            Result directory.
      --quiet               Disable experiment logs and progress bars.
      --override HYDRA_OVERRIDE
                            Additional Hydra override; repeat for multiple overrides.





The agent and environment parameters used for benchmarking the agents on an environment are located in

::

    cfg/env/*

The Hydra launcher profiles are located in:

::

    cfg/profile/*

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

Multiple environments and algorithms can be listed on the same command line. Hydra Submitit submits the resulting
environment, algorithm and seed combinations as a SLURM job array.

Create Plots
------------

If you need to create the plots for a benchmarking folder, you can call the following script

.. code:: shell

    $ ./create_plots.py -d <BenchmarkDir>

where `BenchmarkDir` is the directory of your benchmark, e.g. "logs/benchmark"
