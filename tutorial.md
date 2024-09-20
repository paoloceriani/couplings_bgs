Introduction
============

As described in the README.md file, this tutorial provides details on the functions required to simulate the results presented in Section 5,6,7 and 8 of the paper. 
The python source file can be found in  [`functions.py`](https://github.com/paoloceriani/couplings_bgs/functions.py) for Gaussian CREMS and  [`functions_MH.py`](https://github.com/paoloceriani/couplings_bgs/functions_MH.py) for non Gaussian CREMS.

### `Gaussian`
Consider Model 1 of the paper


$$ y_n | \mu, \mathbf{a}, \tau_0 \sim N\left(\mu +\sum_{k=1}^K a_{i_{k}[n]}^{(k)},\tau_0^{-1}\right),  \quad  n=1,...,N$$

$$a^{(k)}_i | \tau_k \sim N(0, \tau_k^{-1}), \quad   i=1,..., I_k, \, k=1,..., K  $$

$$p(\tau_k) \propto \tau_k^{-0.5}, \quad \text{ for } k=0,..., K  $$

$$p(\mu) \propto 1.$$

The function below allows to generate data from asymptotic regimes 1,2 and 3 and directly run the coupled chains. 
Allows the specification of different number of levels per factor, and different type of algorithms.
The function is called
```
run_experiment(K,I, J,tau_e, tau_k, eps, reg_num, rand,export_results ,T_max, filename, collapsed , variance )
```

**input**
Must specifiy number of factors $K$, levels per factor $I$, precision parameters for generating data $\boldsymbol{\tau}=(\tau_0, \tau_1,..., \tau_k$), $\varepsilon$ parameter of the two-step algorithm. Then
- ```I```: list of values of $I$ for which one wants to run the experiment, e.g. ```I=[10,100,500]``` will run the experiment 3 times with different I
- ```J```: number of repetition of experiment per model specification, must be an integer
- ```reg_num```: $\in\{1,2,3\}$type of asymptotic regime 
- ```rand```: ```True``` o ```False``` random or not initialization of the markov chains 
- ```export_results```: ```True``` o ```False``` for saving results in a file
- ```filename```: string containg path where to save the file
- ```T_max```: maximum number of iteration for each chain 
- ```collapsed```: list of  ```True``` o ```False``` for implementing collapsed or nor version of the algorithm, e.g. ```[True, True, False]``` will run the experiment two times with collapsed algorithm, one with vanilla 
- ```variance```: list of  ```True``` o ```False``` for implementing model with free or fixed variances, must have same length of collapsed 

**output**
- file containing the meeting time of the chains
- ```-1```
**example code**
Import necessary packages
```
import funct_simulation as funct
import numpy as np
import matplotlib.pyplot as plt
```

Define model parameters
```
K= 2
I= np.array([50,100,250,500])
J= 100
tau_e= 1
tau_k= np.ones(K)
eps= 0.1 
reg_num=2
rand=True
```

Function specification
```
export_results = False
T_max = 1000 
filename="output/check_reg2_k2_pv.csv"
collapsed = [True]
variance = [True]

```

Run the funcion
```
funct.run_experiment(K,I, J,tau_e, tau_k, eps, reg_num, rand,export_results ,T_max, filename, collapsed , variance )
```



**plot**

### `Non Gaussian`
