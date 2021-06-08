Classic Control Environments Benchmarks
=======================================


Segway
------


===============  ======
Run Parameters
-----------------------
n_runs           25
n_epochs         50
n_episodes       100
n_episodes_test  10
===============  ======


.. container:: twocol

    .. container:: leftside

        .. highlight:: yaml
        .. literalinclude:: ../../../../results/params/Segway.yaml
            :lines: 3-

    .. container:: rightside

        .. image:: ../../../../results/plots/Segway/J.png
           :width: 400
        .. image:: ../../../../results/plots/Segway/R.png
           :width: 400


LQR
------

===============  ======
Run Parameters
-----------------------
n_runs           25
n_epochs         100
n_episodes       100
n_episodes_test  10
===============  ======



.. container:: twocol

    .. container:: leftside

        .. highlight:: yaml
        .. literalinclude:: ../../../../results/params/LQR.yaml
            :lines: 3-

    .. container:: rightside

        .. image:: ../../../../results/plots/LQR/J.png
           :width: 400
        .. image:: ../../../../results/plots/LQR/R.png
           :width: 400
