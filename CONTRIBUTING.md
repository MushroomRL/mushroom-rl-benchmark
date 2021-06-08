Contributing to MushroomRL Benchmarking Suite
=============================================
We strongly encourage researchers to provide us feedback and contributing
to the MushroomRL Benchmarking Suite. You can contribute in the following ways:
* providing bug reports;
* implementing new state-of-the-art algorithms.

How to report bugs
------------------
Please use the GitHub issues and use the "bug" tag to label it. It is desirable if you can provide a minimal Python script
where the bug occurs. If the bug is confirmed, you can also provide a pull request to fix it, or wait for the maintainers to
resolve the issue.

Implementing new benchmarks
---------------------------
Customized benchmarks can be implemented adding a configuration file and a builder for,
respectively, the environment and algorithm at hand. Configuration files should be added in
``cfg/env/`` in the form of a .yaml file, where the hyper-parameters of the experiment, the
environment, and the algorithms used, are specified. The algorithm description has to be provided
in ``mushroom_rl_benchmark/builders``. A builder class should implement a constructor,
a ``build`` function where the algorithm is created, and a ``default`` function where default
settings for the algorithms are specified.
