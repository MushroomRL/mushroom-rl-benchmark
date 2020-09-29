# MushroomRL Benchmark

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

    usage: benchmark.py [-h] -e ENV [-s] [-t] [-f]

    optional arguments:
    -h, --help         show this help message and exit

    benchmark_script:
    -e ENV, --env ENV  Select an environment for the benchmark.
    -s, --slurm        Flag to signalize the usage of SLURM.
    -t, --test         Flag go signalize that you want to test the script and
                        NOT execute the benchmark.
    -f, --full         Flag to indicate that you want to run the full benchmark.

The agent and environment parameters used for benchmarking the agents on an environment are located in

    benchmark_scripts/env/*

The suite and run parameters used for local, slurm and full_slurm benchmarks are located in

    benchmark_scripts/params_local.yaml
    benchmark_scripts/params_slurm.yaml
    benchmark_scripts/params_slurm_full.yaml

### Execute locally

To run locally call the script like this:

    $ python benchmark_scripts/benchmark.py -e <EnvironmentName> [-t]

### Execute on the cluster

To run locally call the script like this:

    $ python benchmark_scripts/benchmark.py -e <EnvironmentName> -s [-t]

### Execute the full benchmark on the cluster

To run locally call the script like this:

    $ python benchmark_scripts/benchmark.py -e <EnvironmentName> -s -f [-t]
