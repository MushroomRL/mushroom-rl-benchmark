# MushroomRL Benchmark [![Documentation Status](https://readthedocs.org/projects/mushroom-rl-benchmark/badge/?version=latest)](https://mushroom-rl-benchmark.readthedocs.io/en/latest/?badge=latest)


## What is MushroomRL Benchmark?

MushroomRL Benchmark is a benchmarking framework that aims to provide the RL research community with a powerful but easy to use framework to design, execute and present scientifically sound experimentsfor deep RL algorithms. The benchmarking framework builds on top of MushroomRL and utilizes the wide number of algorithms and environments that MushroomRL provides. 

## How to run a benchmark?

A benchmark can be executed like shown in the following code snippet. `XYZBuilder` is an interchangable AgentBuilder. To benchmark a custom algorithm, simply create a new AgentBuilder for you algorithm implementation.

```python
# Initializing of AgentBuilder, 
# EnvironmentBuilder and BenchmarkLogger
logger = BenchmarkLogger()
agent_builder = XYZBuilder.default(...)
env_builder = EnvironmentBuilder(
    env_name, 
    env_params)

# Initializing of the BenchmarkExperiment
exp = BenchmarkExperiment(
    agent_builder, 
    env_builder, 
    logger)

# Running the experiment
exp.run(...)
```

## Installation

You can do a minimal installation of ``mushroom_rl_benchmark`` with:

    pip install  -e .

## Running Examples

You can run the example scripts with:

    python examples/benchmarking_trpo.py

## Execution instructions for Full Benchmark

### Requirements

Go to an python environment, where you have mushroom-rl and mushroom-rl-benchmark installed. 

The script for starting the benchmarks takes following arguments:

    usage: benchmark.py [-h] -e ENV [-s] [-t] [-r]
    
    optional arguments:
      -h, --help         show this help message and exit
    
    benchmark parameters:
      -e ENV, --env ENV  Environment to benchmark.
      -s, --slurm        Flag to use of SLURM.
      -t, --test         Flag to test the script and NOT execute the benchmark.
      -r, --reduced      Flag to run a reduced version of the benchmark.


The agent and environment parameters used for benchmarking the agents on an environment are located in

    cfg/env/*

The suite and run parameters used for local, slurm and full_slurm benchmarks are located in

    cfg/params_local_reduced.yaml
    cfg/params_slurm_reduced.yaml
    cfg/params_local.yaml
    cfg/params_slurm.yaml

### Execute reduced benchmark locally

To run a reduced benchmark locally call the script like this:

    $ ./benchmark.py -e <EnvironmentName> -r

### Execute reduced benchmark on the cluster

To run a reduced benchmark on the cluster call the script like this:

    $ ./benchmark.py -e <EnvironmentName> -s -r

### Execute the full benchmark on the cluster

To run locally call the script like this:

    $ ./benchmark.py -e <EnvironmentName> -s
