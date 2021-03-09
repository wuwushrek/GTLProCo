| Title      | Probabilistic Control of Heteregeneous Swarms Subject to Graph Temporal Logic Specifications: A Decentralized and Scalable Algorithm                  |
|------------|----------------------------------------------------------------------------------------------|
| Authors    | Franck Djeumou, Zhe Xu, and Ufuk Topcu                                                |
| Journal |  IEEE Transactions on Automatic Control                                                            |

## Abstract

We develop a probabilistic control algorithm, GTLProCo, for swarm of heterogeneous agents subject to high-level task specifications. The resulting algorithm not only achieves decentralized control of the swarm but also significantly improves scalability over state-of-the-art existing algorithms. Specifically, we study a setting in which the agents move along the nodes of a graph, and the high-level task specifications for the swarm are expressed in a recently proposed language called graph temporal logic (GTL). By constraining the distribution of the swarm over the nodes of the graph, GTL specifies a wide range of properties, including safety, progress, and response. GTLProCo, agnostic to the number of agents comprising the swarm, controls in a decentralized and probabilistic manner a collective property of the swarm: its density distribution. To this end, it synthesizes a time-varying Markov chain modeling the time evolution of the density distribution under the GTL constraints. We first identify a subset of GTL for which the synthesis of such Markov chain can be formulated as a linear program. Then, in the general case, we formulate the synthesis of such a Markov chain as a mixed-integer nonlinear program (MINLP). We exploit the structure of the problem to provide an efficient sequential mixed-integer linear programming scheme with trust region to approximate the solutions of the MINLP. We evaluate the algorithm in several scenarios, including a search and rescue mission in a high-fidelity ROS-Gazebo simulation, and a disease control case study.

## Dependencies

In a virtual environment (e.g., conda) with Python 3, this package requires ``numpy``, ``scipy``, ``sympy``, and ``matplotlib`` as follows:
```
python -m pip install numpy scipy sympy matplotlib
```

For comparison with off-the-shelf and efficient MINLP solvers, this package can be used to solve the MINLP problem through ``couenne``, ``bonmin``. The binaries for these libraries can be obtained directly from the [AMPL website](https://ampl.com/products/solvers/open-source). Then, install ``pyomo`` [the python interface to use these solvers](https://pyomo.readthedocs.io/en/stable/installation.html).
```
python -m pip install pyomo
```

In addition, we can also use [SCIP](https://www.scipopt.org/index.php#download) ``one of the fastest non-commercial solvers for mixed integer programming (MIP) and mixed integer nonlinear programming (MINLP)``. To this end, install the latest SCIP package by following the instruction in the README file of the installer available on [SCIP website](https://www.scipopt.org/index.php#download). Then, the [python interface](https://github.com/scipopt/PySCIPOpt/blob/master/INSTALL.md) can be found and installed.
```
# Make sure [SCIPOPTDIR] is set to the right path
python -m pip install pyscipopt
``` 

## More dependencies
Finally, our sequential MILP solver with trust region requires Gurobi to solve each subproblems. Install gurobi and the python interface as specified in the [gurobi website](https://www.gurobi.com/documentation/9.1/quickstart_linux/index.html).

## Usage