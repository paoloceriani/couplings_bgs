# couplings_bgs
Python code associated to the article "Linear-cost unbiased posterior estimates for crossed effects and matrix factorization models via couplings". Key contribution of the paper is, as outlined in the abstract:
> We design and analyze unbiased MCMC schemes based on couplings of blocked Gibbs samplers (BGS), whose total computational cost scales linearly with the number of param- eters and data points.

The repository contains **code** and **tutorials** for implementing such a coupling procedure. 
The source file [`functions.py`](https://github.com/paoloceriani/couplings_bgs/functions.py) contains the building blocks for implementing couplings of Gaussian BGS for Model 1 in the paper.

The source file [`funct_simulation_MH.py`](https://github.com/paoloceriani/couplings_bgs/funct_simulation_MH.py) contains the building blocks for implementing couplings of non Gaussian BGS for Model 2 in the paper with laplace response.

In [`tutorial.md`](https://github.com/paoloceriani/couplings_bgs/tutorial.md) we report some example code and basic explanations to reproduce to code for the simulations reported in the paper. 

