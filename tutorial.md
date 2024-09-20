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

## Code 
**Input**
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

**Output**
- file containing the meeting time of the chains
- ```-1```
**Example code**
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



**Plot**
In order to obtain the plots in the paper we have implemented the following ad hoc function
```
def plot_results(file_name, K,tau_e, tau_k, delta, eps,reg_num,rand=True,vanilla = False, save= False,S=1, rho=1, cappa = 1, output_name =[], skip = False)
```

**Input**
- ```file_name```: path to the file containing the results of the experiment
-  ```K,tau_e, tau_k, delta, eps,reg_num```: model specification as before
-``` rand```: ```True``` o ```False``` for initialization 
- ```vanilla```: ```True``` for vanilla ```False``` for collapsed
- ```save```: ```True``` for saving the output picture  ```False``` otherwise
-```output_name```: string containing the name of output

**Output**
- plot of the average meeting times and the relative bound
- .png file containg the picture if ```save==True```

**Example code**
Continuing the one from before

```
reg_num = 1
file_name= "output/pv_reg1_k2.csv" 
delta= [0]
eps=0.1
rand=True
vanilla = True
save= True
output_name = "/Users/Family/Desktop/nat_scale_reg1_k2_pv"
plot_results(file_name, K,tau_e, tau_k, delta, eps,reg_num,rand,vanilla, save, output_name = output_name)
```


### `Non Gaussian`
Consider Model 2 of the paper with

	$$\mathcal{L}(y_n | \mu, \mathbf{a} ) = \mathcal{L}(y_n | \eta_n ),\text{ with }\eta_n = \mu +\sum_{k=1}^K a_{i_{k}[n]}^{(k)}  \quad\text{ for } n=1,...,N, $$
 $$\mathcal{L}(y_n | \mu, \mathbf{a} )= Lapl( \eta_n ,1) $$

## Code 
We report below the code to reproduce our simulations

Import necessary modules 

```
import funct_simulation_MH as fmh
import numpy as np
```

Model and parameter specification 

```
num = 2
K = 2
I = [100,200,500]
J = 100
#code implemented for L=1
L = 1
T_max = 1000
S = [1]
tau_k = np.ones(K)
b = 1
```

Auxiliary arrays to keep track of the results
```
times = np.zeros((len(S ), len(I), J))
#try to estimate only the variance tau[0], as check
estim = np.zeros((len(S ), len(I), J))
true = np.zeros((len(S ), len(I), J))
```

Run experiments for different ```I,S,J```

```
for en,i in enumerate(I):
    print('#####################  '+ str(i)+ '  #################################################################\n' )
    data,a = fmh.asymptotic_regimes_lapl(num,K,I=10,b=1,mu=0, tau_k=tau_k, return_a = True)
    print("go!")
    for esse,s in enumerate(S):
        for j in range(J):
            iterv = fmh.iter_value_laplace(data, T_max,a,0,  0, rand = True)
            iterv_2 = fmh.iter_value_laplace(data, T_max,a,0,  0, rand = True)
            mask = np.array([np.array([data.ii[:,k] == i for i in range(data.iss[k])]) for k in range(K)])
            for l in range(L):
                iterv.update(data,l,mask,S=s, fixed = False)
            dist = iterv.distance(iterv_2)
            #print(dist/(K*(i+1)) )
            if dist/(K*(i+1)) < 10:
                close = True
            else:
                close = False
            for t in (range(T_max)):
                if iterv.coupled_update(iterv_2, data,t,L,mask,b,s, fixed = False, close = close, factorized_proposals = True, factorized_acceptance = True, a_pvb= True, a_pvf = True) == -1:
                    break
            times[esse,en,j]= t+1
            true[esse,en,j] = iterv.tau[0]
            estim[esse,en,j]= np.mean(iterv.tau_chain[0,(t+1)//2: t+1] + (iterv.tau_chain[0,(t+1)//2+1: t+2] - iterv_2.tau_chain[0,(t+1)//2+1: t+2] ))
```

Plot the results 
```
save = True
output_name = "MH_fact"
plt.figure(figsize=(10,5))
font = {'fontname':'Cambria Math'}
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.xticks(range(len(I)), np.array(I)*K+1)
plt.plot(range(len(I)), np.mean(times[0], axis = 1), label = "LC MH RW fact, S="+str(S[0]))
plt.scatter(range(len(I)), np.mean(times[0], axis = 1))
plt.plot(range(len(I)), np.mean(times[1], axis = 1), label = "LC MH RW fact, S="+str(S[1]))
plt.scatter(range(len(I)), np.mean(times[1], axis = 1))
plt.ylabel("Meeting time", fontsize=30  , **font)
plt.xlabel("Parameters number", fontsize= 30, **font)
plt.legend(fontsize=25)
plt.rc('font',family='Cambria Math')
matplotlib.rc('font',family='Cambria Math')
#plt.yscale("log")
plt.title("Average meeting times",fontsize = 30,**font)
plt.ylim(0)
plt.grid(True, which="both")
if save:
    plt.savefig(str(output_name)+'.png', bbox_inches="tight")
plt.show()
```
            
