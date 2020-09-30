====================
MushroomRL Benchmark
====================

Reinforcement Learning python library
-------------------------------------

.. highlight:: python

MushroomRL Benchmark is a benchmarking tool for the Mushroom RL library.
The focus of this benchmarking tool is  to benchmark the results of deep reinforcement
learning algorithms, in particular Deep Actor-Critic.
The idea behind MushroomRL Benchmarking is to have a complete platform to run batch
comparisons of Deep RL algorithms implemented in MushroomRL under a set of standard
benchmark tasks.

With MushroomRL Benchmarking you can:

- Run the benchmarks in a local machine, both sequentially and in parallel fashion
- Run experiments on a SLURM-based cluster.


Basic run example
-----------------


Download and installation
-------------------------

MushroomRL Benchmark can be downloaded from the
`GitHub <https://github.com/MushroomRL/mushroom-rl-benchmark>`_ repository.
Installation can be done running

::

    cd mushroom-rl-benchmark
    pip install -e .[all]

To compile the documentation:

::

    cd mushroom-rl-benchmark/docs
    make html

or to compile the pdf version:

::

    cd mushroom-rl-benchmark/docs
    make latexpdf

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Benchmarks Results:
   :glob:

   source/benchmarks/*

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API:
   :glob:

   source/api/*

