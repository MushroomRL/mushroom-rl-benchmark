Contributing to MushroomRL
==========================
We strongly encourage researchers to provide us feedback and contributing
to MushroomRL. We need your ideas to keep improving the library,
and we are happy to merge your work in the library.
You can contribute to MushroomRL in several ways:
* providing bug reports;
* implementing new state-of-the-art algorithms;
* proposing improvements of the library, or documentation.

How to report bugs
------------------
Please use the GitHub issues and use the "bug" tag to label it. It is desirable if you can provide a minimal Python script
where the bug occurs. If the bug is confirmed, you can also provide a pull request to fix it, or wait for the maintainers to
resolve the issue.

Implementing new algorithms
---------------------------
Although each contribution is welcomed, we will only accept high-quality code that will reflect the key ideas
of modularity and flexibility of MushroomRL. Please keep this in mind before submitting
a pull request.

Every algorithm in MushroomRL must extend the ``Agent`` interface. If the algorithm belongs to a class of algorithms already
implemented in MushroomRL, then we expect the author to extend the appropriate interface. If new policies or new parameter types
are needed, please extend the corresponding base classes. Using other MushroomRL modules (e.g. ``Regressor``, ``Features``) is
highly encouraged. Together with the algorithm, you must provide an example and a test case. Every algorithm matching our
standard of code and scientific quality and providing the example and the test, will be merged in MushroomRL. It is encouraged
to include the citation to the paper describing the algorithm in the docstring of the class.

Proposing improvements
----------------------
If you find some lack of features, please open a GitHub issue and use the "enhancement" tag to label it. However, before
opening a pull request, we suggest to discuss it with the maintainers of the library, that will be glad of providing
suggestions and help in case the proposed improvement is particularly interesting.
