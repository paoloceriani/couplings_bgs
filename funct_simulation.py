import numpy as np
import scipy as sp
import scipy.stats
import numpy.random as random
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.linalg as la
import math
from scipy.special import erfinv
import warnings


# code for couplings of univariate/ multivariate gaussians

def normal_dens_1d(mu,sigma,x):
    if np.power(x-mu,2)/sigma**2 > 1e20:
        return 0 
    else:
        return np.power(2*np.pi*sigma**2,-0.5)*np.exp(-0.5*np.power(x-mu,2)/sigma**2)

def normal_dens_d(mu,Sigma,x, diag=False): #here Sigma is var/cov matrix, so no issue 
    if diag==False:
        if 0.5*(x-mu)@la.inv(Sigma)@(x-mu)> 1e15:
            return 0
        else:
            return np.power(2*np.pi*la.det(Sigma),-0.5)*np.exp(-0.5*(x-mu)@la.inv(Sigma)@(x-mu))
    else:
        if ((np.sum(np.power(x-mu,2)/np.diag(Sigma)))> 1e20) or (np.sum(np.log(np.diag(Sigma))) > 1e20):
            return 0
        else: 
            try:    
                return np.exp( (-0.5*np.sum(np.power(x-mu,2)/np.diag(Sigma))) -0.5 *len(mu)*np.log(2*np.pi))*np.power(np.prod(np.diag(Sigma)), -0-5)
            except RuntimeWarning:
                print((np.sum(np.power(x-mu,2)/np.diag(Sigma))), np.sum(np.log(np.diag(Sigma))) , len(mu)*np.log(2*np.pi) )
                return 0
                
def maximal_coupling_normal_1d(mu_x, mu_y, sigma_x, sigma_y):
    x = np.random.normal(mu_x,sigma_x)
    w = np.random.rand()*normal_dens_1d(mu_x,sigma_x,x)
    
    if w < normal_dens_1d(mu_y,sigma_y,x):
        return x,x
    else:
        while True:
            y = np.random.normal(mu_y, sigma_y)
            w = np.random.rand()*normal_dens_1d(mu_y,sigma_y,y)
            if w > normal_dens_1d(mu_x,sigma_x,y):
                return x,y
                

def maximal_coupling_normal_d(mu_x, mu_y, Sigma_x, Sigma_y,diag=False):
    x = np.random.multivariate_normal(mu_x,Sigma_x)
    w = np.random.rand()
    
    
    # np.power(2*np.pi*la.det(Sigma),-0.5)*np.exp(-0.5*(x-mu)@la.inv(Sigma)@(x-mu))
    if diag == False:
        if np.log(w) < -0.5*np.log(la.det(Sigma_y)/la.det(Sigma_x))-0.5*( (x-mu_y)@la.inv(Sigma_y)@(x-mu_y)- (x-mu_x)@la.inv(Sigma_x)@(x-mu_x)):
            return x,x
        else:
            while True:
                y = np.random.multivariate_normal(mu_y, Sigma_y)
                w = np.random.rand()
                if np.log(w) > -0.5*np.log(la.det(Sigma_x)/la.det(Sigma_y))-0.5*( (y-mu_x)@la.inv(Sigma_x)@(y-mu_x)- (y-mu_y)@la.inv(Sigma_y)@(y-mu_y)):
                    return x,y  
    else:
        if np.log(w) < -0.5*np.sum(np.log(np.diag(Sigma_y)/np.diag(Sigma_x))) -0.5*( np.sum((x-mu_y)**2/np.diag(Sigma_y))- np.sum((x-mu_x)**2/np.diag(Sigma_x))) :
            return x,x
        else:
            while True:
                y = np.random.multivariate_normal(mu_y, Sigma_y)
                w = np.random.rand()
                if np.log(w) >  -0.5*np.sum(np.log(np.diag(Sigma_x)/np.diag(Sigma_y))) -0.5*( np.sum((y-mu_x)**2/np.diag(Sigma_x))- np.sum((y-mu_y)**2/np.diag(Sigma_y))):
                    return x,y  
          
                

def reflection_coupling_1d(mu_1,mu_2, sigma):
    x = np.random.normal(0,1)
    u = np.random.rand()
    z = (mu_1-mu_2)/sigma
    if u < np.exp(-0.5*(z**2+ 2*z*x)):
        y=x+z
    else:
        y = -x
    x = mu_1 + sigma*x
    y = mu_2 + sigma*y
    return x,y

def reflection_coupling(mu_1, mu_2, Sigma, diag =False):
    if (mu_1==mu_2).all():
        X= np.random.multivariate_normal(mu_1, Sigma)
        return X,X
    X = np.random.normal(0,1, size = len(mu_1)) #they are independent 
    u = np.random.rand()
    if diag==False:
        L = la.cholesky(Sigma)
        z = la.inv(L)@(mu_1-mu_2)
        e = z/la.norm(z)
    else:
        #L= np.sqrt(Sigma[0,0])*np.identity(len(mu_1))
        L= np.sqrt(Sigma)
        z = (mu_1-mu_2)/np.diag(L)
        e = (mu_1-mu_2)/la.norm(mu_1-mu_2)
        
    if u< np.exp(-0.5*(z@z+2*z@X)):
        Y=X+z
    else:
        Y=X-2*(e@X)*e
    X = mu_1+ L@X
    Y = mu_2+ L@Y
    return X,Y
    
# we do maximal coupling on gamma with same shape parameter
def maximal_coupling_gamma(a, b_x, b_y):
    x = np.random.gamma(a, b_x)
    w = np.random.rand()
    
    if np.log(w) < (a*np.log(b_x/b_y)-x/b_y+x/b_x):
    # if np.log(w) < np.log(b_x/b_y)*a*(x*(1/b_x-1/b_y)):
        return (x,x)
    else:
        while True:
            y = np.random.gamma(a, b_y)
            w = np.random.rand()
            #if np.log(w) > np.log(b_y/b_x)*a*(y*(1/b_y-1/b_x)):
            if np.log(w) > (a*np.log(b_y/b_x)-y/b_x + y/b_y ):
            #if np.log(w) < (a*np.log(b_y/b_x)-y/b_y+y/b_x):
                return (x,y)
                

def d_tv_d(mu_x, mu_y, Sigma):
    return erf(np.sqrt((mu_x-mu_y)@la.inv(Sigma)@(mu_x-mu_y)/8))
def d_tv_1d(mu_x, mu_y, sigma):
    return erf(np.abs(mu_x-mu_y)/(sigma*2*np.sqrt(2)))
    
# auxiliary functions for Crossed random effects models

def convert_col(x):
    return x.replace({i:e for e,i in enumerate(set(x))})
 

def countmemb(itr,I):
    # in this way itr might have different dimensions for different factor, no good
    count = np.zeros((itr.shape[1],I))
    for d in range(itr.shape[1]):
        for val in itr[:,d]:
            count[d,val] += 1
    return count
 
 
def sum_on_all(a, N_sl, k, N_s, iss):
    #limitiamo il numero di liste 
    K = len(a)
    res = np.zeros(iss[k])
    kr = np.array(range(K))
    for l in kr[kr!= k]:
        res += N_sl[k][l]@a[l]
    
    for j in range(iss[k]):
        if N_s[k][j] != 0:
            res[j] = res[j]/ N_s[k][j]
        else:
            res[j] = 0

    return res


# Python class with all the functions needed for generate data from Gaussian CREMS and sampling from their posteriors
class Data:
    def __init__(self):
        self.K = 0
        self.I = 0
        self.N = 0
        self.mu = 0
        self.ii = None
        self.iss = None
        self.mean_y = None
        self.mean_y_s = None
        self.N_s = None
        self.N_sl = None

    #generate data from the model, assign N oberservations randomly to each of the I factor of level K
    def generate(self, N,K,I,mu):
        self.K = K
        self.I = I
        self.N = N
        self.mu = mu
        # supposing uniform assignment of individuals to factor/levels   
        self.ii = np.random.randint(0,self.I, size=(self.N,self.K)) # this should go from 0 to I-1
        self.iss = self.I*np.ones(self.K, dtype = int) 
        self.a = [np.random.normal(0,1,size=I) for k in range(K)]
        aux = np.array([np.sum([self.a[k][self.ii[n,k]] for k in range(K)]) for n in range(N)])
        self.y = aux+self.mu+np.random.normal(0,1, size = self.N) 
        self.mean_y = np.mean(self.y)
        self.mean_y_s = np.array([np.array([np.mean(self.y[self.ii[:,k] == h]) if not np.isnan(np.mean(self.y[self.ii[:,k] == h])) else 0 for h in range(0,self.iss[k])]) for k in range(self.K) ])
        self.N_s = countmemb(self.ii,self.I)
        self.N_sl = [[ np.zeros((self.iss[i],self.iss[j])) for j in range(self.K)] for i in range(self.K)]
        for i in range(self.K) :
            for j in range(self.K):
                for n in range(self.N):
                    self.N_sl[i][j][self.ii[n,i], self.ii[n,j]] += 1

    # function needed for real data, to cast in the same format as above
    def import_df(self,df, col_name):
        self.y = np.array(df['y'])
        self.ii= np.array(df.loc[:, col_name].apply(convert_col, axis = 0)).astype('int')
        self.I = self.ii.max()+1
        self.K = self.ii.shape[1]
        self.N= self.ii.shape[0]
        self.iss = np.max(self.ii, axis = 0).astype('int')+1 
        self.mean_y = np.mean(self.y)
        self.mean_y_s = [np.array([np.mean(self.y[self.ii[:,k] == h]) if not np.isnan(np.mean(self.y[self.ii[:,k] == h])) else 0 for h in range(0,self.iss[k])]) for k in range(self.K) ]
        
        self.N_s = [np.zeros(self.iss[k]) for k in range(self.K)]
        for d in range(self.K):
            for val in self.ii[:,d]:
                self.N_s[d][val] += 1
        
        self.N_sl = [[ np.zeros((self.iss[i],self.iss[j])) for j in range(self.K)] for i in range(self.K)]
        for i in range(self.K) :
            for j in range(self.K):
                for n in range(self.N):
                    self.N_sl[i][j][self.ii[n,i], self.ii[n,j]] += 1
                    
    # function to compute the theoretical convergence rate ecploiting the work in https://www.jstor.org/stable/2346048
    # both for plain vanilla and collapsed gibbs
    def conv_rate(self, tau_e, tau):
        index = np.zeros(1+self.K).astype(int)
        index[0] = 1
        index[1:] = (np.cumsum(self.iss)+1).astype(int)
        Q=  np.zeros((1+np.sum(self.iss), 1+np.sum(self.iss)))
        Q[0,0] = self.N*tau_e
        Q[0,1:] = tau_e*np.array([j for i in self.N_s for j in i])
        Q[1:, 0] = tau_e*np.array([j for i in self.N_s for j in i])
        for i in range(self.K):
            for j in range(self.K):
                if i == j:                        
                    Q[index[i]:index[i+1],index[j]:index[j+1]] = self.N_sl[i][i]+tau[i]*np.identity(self.iss[i])
                else:
                    Q[index[i]:index[i+1],index[j]:index[j+1]] = self.N_sl[i][j]*tau_e
        A = np.identity(Q.shape[0])- la.block_diag(1/Q[0,0],*[la.inv(Q[index[i]:index[i+1],index[i]:index[i+1]]) for i in range(self.K)])@Q
        U = la.triu(A)
        L = la.tril(A)
        
        # vanilla scheme 
        B_pv = la.inv((np.identity(A.shape[0])-L))@U
        Sigma = la.inv(Q)
        rho_pv = np.max(np.abs(la.eigvals(B_pv)))


        # collapsed
        index = np.zeros(self.K+1).astype(int)
        index[0] = 0
        index[1:] = (np.cumsum(self.iss)).astype(int)
        D = Q[1:,1:]
        Q_c= D-Q[1:,0].reshape(np.sum(self.iss),1)@Q[0,1:].reshape(1,np.sum(self.iss))/self.N
        A = np.identity(Q_c.shape[0])- la.block_diag(*[la.inv(Q_c[index[i]:index[i+1],index[i]:index[i+1]]) for i in range(self.K)])@Q_c
        
        U = la.triu(A)
        L = la.tril(A)
        B_coll = la.inv((np.identity(A.shape[0])-L))@U
        rho_coll = np.max(np.abs(la.eigvals(B_coll)))
        return rho_pv, rho_coll,Sigma, B_pv, B_coll

def distance(val_1,val_2):
# compute square distances between vector of ALL parameters    
    return  (val_1.tau-val_2.tau)@(val_1.tau-val_2.tau) + (val_1.tau_e-val_2.tau_e)**2 + (val_1.mu-val_2.mu)**2+ np.sum([np.sum(np.power(val_1.a[k]-val_2.a[k],2)) for k in range(len(val_1.a))])


# class of the values of the MCMC chains, for Model 1 with fixed or free variance
class iter_value:    
    def __init__(self, data,T, rand= False, var_fixed = False, var = 9):
        
        self.var_fixed = var_fixed
        if rand:
            if not var_fixed:
                self.tau = np.random.gamma(1/2,2, size= data.K)
                self.tau_e = np.random.gamma(1/2,2)
            else:
                self.tau = np.ones(data.K)
                self.tau_e = 1.0
            self.mu = np.random.normal(0,np.sqrt(var))
            self.a = [np.random.normal(0,np.sqrt(var),size=data.iss[k]) for k in range(data.K)]
            
        else:
            self.tau = np.ones(data.K)
            self.tau_e = 1.0
            self.mu = np.random.normal(0,np.sqrt(1/self.tau_e))
            self.a = [np.zeros(data.iss[k]) for k in range(data.K)] #chain 
            #chain values
            
        self.a_means = np.array([sum(self.a[k] * data.N_s[k])/data.N for k in range(data.K)])
        self.SS0 = 0
        
        self.SS0_chain = np.zeros(T+1)
        self.a_means_chains= np.zeros((data.K,T+1 ))
        self.mu_chain = np.zeros(data.K*T+1)
        self.tau_e_chain = np.zeros(T+1)
        self.tau_chain = np.zeros((data.K,T+1 ))
        self.mu_chain[0]= self.mu
        self.a_means_chains[:,0] = self.a_means
        self.tau_e_chain[0] = self.tau_e
        self.tau_chain[:,0] = self.tau
        
            
    def sum_on_all(self, data,k):
            K = len(self.a)
            res = np.zeros(data.iss[k])
            kr = np.arange(K)
            for l in kr[kr!=k]:
                res += data.N_sl[k][l]@self.a[l]
            for j in range(data.iss[k]):
                if data.N_s[k][j] != 0:
                    res[j] = res[j]/ data.N_s[k][j]
                else:
                    res[j] = 0
            return res

    # on step of the vanilla Gibbs sampling, if collapsed =True, implement collapsed gibbs
    
    def update(self,data, t, collapsed, PX):
        pred = np.zeros(data.N)
        if not collapsed: #update mu given a^(0:K)
            mu_mean = data.mean_y - np.sum(self.a_means)
            mu_var = 1/(data.N*self.tau_e)
            self.mu = np.random.normal(mu_mean,  np.sqrt(mu_var))
            
        for k in range(data.K):
            s_k = data.N_s[k]*self.tau_e / (self.tau[k]+ data.N_s[k]*self.tau_e)    
            sum_a_l=self.sum_on_all(data,k)
            if collapsed:
                mu_mean= (s_k@(data.mean_y_s[k] - sum_a_l))/sum(s_k)
                var_mean = 1/(self.tau[k]*sum(s_k))
                aux = np.random.normal(0,1)
                self.mu = mu_mean+aux*np.sqrt(var_mean)
                self.mu_chain[1+t*data.K+k]=self.mu


            a_k_mean = (data.mean_y_s[k]-self.mu-sum_a_l)/(1+self.tau[k]/ (self.tau_e*data.N_s[k]))
            a_k_var = 1/(data.N_s[k]*self.tau_e+self.tau[k])  
            aux = np.random.normal(0,1, size= data.iss[k])
            self.a[k]= a_k_mean+ np.sqrt(a_k_var)*aux
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            pred += self.a[k][data.ii[:,k]]
            
            if PX :
                prec_alpha_k= self.tau_e*sum(data.N_s[k]*np.power(self.a[k],2))
                mean_alpha_k = sum(self.a[k]*(data.N_s[k]*self.tau_e+self.tau[k])*a_k_mean)/prec_alpha_k
                aux = np.random.normal(0,1)
                alpha_k = mean_alpha_k+ aux/np.sqrt(prec_alpha_k)
                self.a[k] = self.a[k]*alpha_k

            if self.var_fixed == False:
                self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/sum(np.power(self.a[k],2)))

        pred += self.mu
        self.SS0 = np.sum(np.power((data.y-pred),2))

        if self.var_fixed == False:
            self.tau_e = np.random.gamma(size=1, shape= data.N/2-0.5, scale=2/self.SS0)
        
        
        self.a_means_chains[:, t+1] = self.a_means
        self.SS0_chain[t+1] = self.SS0
        if not collapsed:
            self.mu_chain[t+1]=self.mu
        
        self.tau_e_chain[t+1] = self.tau_e
        self.tau_chain[:, t+1] = self.tau

    # implement one iteration of the collapsed Gibbs, val_2 refers to the coupled chain and belongs to iter_value class
    def coupled_update(self, val_2,data,t,l, collapsed, PX, close):
        pred = 0
        pred_2 = 0
        aux_r = np.random.randint(1,5000, size = data.K+1)
        if not collapsed: #update mu given a^(0:K)
            mu_mean = data.mean_y - np.sum(self.a_means)
            mu_var = 1/(data.N*self.tau_e)
            mu_mean_2 = data.mean_y - np.sum(val_2.a_means)
            mu_var_2 = 1/(data.N*val_2.tau_e)
            
            if not close:
                aux = np.random.normal(0,1)
                self.mu = mu_mean+ aux*np.sqrt(mu_var)
                val_2.mu= mu_mean_2+ aux*np.sqrt(mu_var_2)
            else:
                self.mu,val_2.mu = maximal_coupling_normal_1d(mu_mean, mu_mean_2,np.sqrt(mu_var), np.sqrt(mu_var_2))
          

        for k in range(data.K):
            s_k =   data.N_s[k]*self.tau_e / (self.tau[k]+ data.N_s[k]*self.tau_e)
            sum_a_l =  self.sum_on_all(data,k)
            s_k_2 = data.N_s[k]*val_2.tau_e / (val_2.tau[k]+ data.N_s[k]*val_2.tau_e)
            sum_a_l_2 =  val_2.sum_on_all(data,k)
            
            if self.var_fixed == False:
                if not close:
                    np.random.seed(aux_r[k])
                    self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/sum(np.power(self.a[k],2)))
                    np.random.seed(aux_r[k])
                    val_2.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/sum(np.power(val_2.a[k],2)))
                else:
                    self.tau[k], val_2.tau[k] = maximal_coupling_gamma(data.iss[k]/2-0.5,2/sum(np.power(self.a[k],2)),2/sum(np.power(val_2.a[k],2)))
                
            if collapsed:
                mu_mean= sum( s_k *(data.mean_y_s[k] - sum_a_l))/sum(s_k)
                mu_mean_2= sum( s_k_2 *(data.mean_y_s[k] - sum_a_l_2))/sum(s_k_2)
                var_mean = 1/(self.tau[k]*sum(s_k))
                var_mean_2 = 1/(val_2.tau[k]*sum(s_k_2))
                
                if not close:
                    aux = np.random.normal(0,1)
                    self.mu = mu_mean+aux*np.sqrt(var_mean)
                    val_2.mu = mu_mean_2+aux*np.sqrt(var_mean_2)
                else:
                    self.mu, val_2.mu = maximal_coupling_normal_1d(mu_mean, mu_mean_2, np.sqrt(var_mean), np.sqrt(var_mean_2))
            a_k_mean = self.tau_e*data.N_s[k]/(self.tau_e*data.N_s[k]+self.tau[k])*(np.array(data.mean_y_s[k])-self.mu-sum_a_l)
            a_k_var = 1/(data.N_s[k]*self.tau_e+self.tau[k])  

            a_k_mean_2 = val_2.tau_e*data.N_s[k]/(val_2.tau_e*data.N_s[k]+val_2.tau[k])*(np.array(data.mean_y_s[k])-val_2.mu-sum_a_l_2)
            a_k_var_2 = 1/(data.N_s[k]*val_2.tau_e+val_2.tau[k])  
            
            if not close:
                aux = np.random.normal(0,1, size= data.iss[k])
                self.a[k]= a_k_mean+ np.sqrt(a_k_var)*aux
                val_2.a[k]= a_k_mean_2+ np.sqrt(a_k_var_2)*aux
                
            else:
                if (self.tau[k]==val_2.tau[k]).all():
                    self.a[k], val_2.a[k] = reflection_coupling(a_k_mean, a_k_mean_2, np.diag(a_k_var),  diag=True)
                else:
                    self.a[k], val_2.a[k] = maximal_coupling_normal_d(a_k_mean, a_k_mean_2, np.diag(a_k_var), np.diag(a_k_var_2),diag = True)

            self.a_means[k] = sum(self.a[k]*data.N_s[k])/data.N
            val_2.a_means[k] = sum(val_2.a[k]*data.N_s[k])/data.N
            
            if PX:
                prec_alpha_k= self.tau_e*sum(data.N_s[k]*np.power(self.a[k],2))
                mean_alpha_k = sum(self.a[k]*(data.N_s[k]*self.tau_e+self.tau[k])*a_k_mean)/prec_alpha_k

                prec_alpha_k_2= val_2.tau_e*sum(data.N_s[k]*np.power(val_2.a[k],2))
                mean_alpha_k_2 = sum(val_2.a[k]*(data.N_s[k]*val_2.tau_e+val_2.tau[k])*a_k_mean_2)/prec_alpha_k_2

                
                if not close:
                    aux = np.random.normal(0,1, size= 1)
                    alpha_k = mean_alpha_k+ aux/np.sqrt(prec_alpha_k)
                    alpha_k_2 = mean_alpha_k_2+ aux/np.sqrt(prec_alpha_k_2)
                else:
                    alpha_k, alpha_k_2 =maximal_coupling_normal_1d(mean_alpha_k, mean_alpha_k_2, 1/np.sqrt(prec_alpha_k),  1/np.sqrt(prec_alpha_k_2))
                     
                self.a[k] = self.a[k]*alpha_k
                val_2.a[k] = val_2.a[k]*alpha_k_2

            pred += self.a[k][data.ii[:,k]]
            pred_2 += val_2.a[k][data.ii[:,k]]
        

        pred += self.mu
        pred_2 += val_2.mu
        
        self.SS0 = sum(np.power((data.y-pred),2))
        val_2.SS0 = sum(np.power((data.y-pred_2),2))
        
        if self.var_fixed == False:
            if not close:
                np.random.seed(aux_r[data.K])
                self.tau_e = np.random.gamma(shape= data.N/2-0.5, scale=2/self.SS0)
                np.random.seed(aux_r[data.K])
                val_2.tau_e = np.random.gamma(shape= data.N/2-0.5, scale=2/val_2.SS0)
            else:
                self.tau_e, val_2.tau_e = maximal_coupling_gamma(data.N/2-0.5,2/self.SS0, 2/val_2.SS0)

        
        self.a_means_chains[:, t+l+1] = self.a_means
        self.SS0_chain[t+l+1] = self.SS0
        self.mu_chain[t+l+1]=self.mu
        self.tau_e_chain[t+l+1] = self.tau_e
        self.tau_chain[:, t+l+1] = self.tau

        val_2.a_means_chains[:, t+1] = val_2.a_means
        val_2.SS0_chain[t+1] = val_2.SS0
        val_2.mu_chain[t+1]=val_2.mu
        val_2.tau_e_chain[t+1] = val_2.tau_e
        val_2.tau_chain[:, t+1] = val_2.tau
         
        if (self.mu == val_2.mu) and np.array([(self.a[k] == val_2.a[k]).all() for k in range(data.K)]).all() and self.tau_e==val_2.tau_e and np.array(self.tau == val_2.tau).all() :         
            return -1
        
    def __str__(self):
        return "Values mu: %f\ntau_e: %f\na:%s\ntau_k %s" %(self.mu, self.tau_e, str(self.a),str(self.tau))


# function that, given model specification and parameters and GENERATED data as input, generate the coupled chains
def MCMC_sampler_coupled(data,collapsed=False, PX=False,L=1, T = 100, dist = 1e-2, rand= False, var_fixed = False, gen_var = 9 ): 
    N = data.N
    K = data.K
    
    val_1 = iter_value(data, T+L+1,rand, var_fixed, gen_var)
    val_2 = iter_value(data, T+1, rand, var_fixed, gen_var)    
    coupling = False
    
    for l in range(L):
        val_1.update(data,l,collapsed, PX)

    original_distance = distance(val_1, val_2) 
    t=0
    while t<T:
        close = (distance(val_1, val_2) < dist )
        if val_1.coupled_update(val_2,  data,t,L,collapsed, PX,close) == -1:
            break
        t+=1
    
    return val_1,val_2,original_distance, t



# marginal sampler
def MCMC_sampler_single(data, T,collapsed=False, PX=False, rand=False, var_fixed = False): 
    N = data.N
    K = data.K
    val_1 = iter_value(data,T, rand, var_fixed)
    t=0
    while t<T:
        val_1.update(data, t,collapsed, PX)
        t+=1
    return val_1

# function that, given model specification and parameters and REAL data as input, generate the coupled chains
def MCMC_sampler_coupled_realdataset(data,collapsed=False, PX=False,L=1, T = 100, dist = 1e-2, rand= False, tau = None, tau_e = None): 
    N = data.N
    K = data.K
   
    if isinstance(tau, (np.ndarray, list)):
        var_fixed=True
        val_1 = iter_value(data, T+L+1,rand, var_fixed)
        val_2 = iter_value(data, T+1, rand, var_fixed)
        np.copyto(val_1.tau, tau)
        np.copyto(val_2.tau, tau )   
        val_1.tau_e = tau_e
        val_2.tau_e = tau_e
    else: 
        val_1 = iter_value(data, T+L+1,rand, var_fixed=False)
        val_2 = iter_value(data, T+1, rand, var_fixed=False)    
    
    coupling = False
    for l in range(L):
        val_1.update(data,l,collapsed, PX)
    t=0
    while t<T:
        close = (distance(val_1, val_2) < dist )
        a = val_1.coupled_update(val_2,  data,t,l,collapsed, PX,close)
        if a == -1:
            print("finished:",t)
            break
        t+=1
    
    return val_1,val_2, t
 
# generate data according to asymptotic regime 1 / 2 / 3
def asymptotic_regimes(num,K,I=10,S=1, rho=1, cappa = 1, return_a = False, mu=0):
    x =Data()
    data = pd.DataFrame()
    
    if num ==1:
        # dense regime
        p = 0.1 / (I**(K-2))
    elif num == 2:
        # sparse 
        p = 10/(I**(K-1))
    elif num == 4:
        p = 0.1/(I**(K-1.5))
    
    N = np.random.binomial(I**K, p ,1)[0]
    iss = [I]*K
    tau = np.ones(K)
    a = [np.random.normal(0,1/np.sqrt(tau[k]),size=iss[k]) for k in range(K)]  
    y= np.zeros(np.sum(N))
    ii = np.zeros((N,K)).astype(int)
        
    xx = random.sample(range(I**K),N) 
    if K == 2:
        for i in range(N):
            ac = xx[i]//I
            b = xx[i]-ac*I
            ii[i,0] = ac
            ii[i,1] = b
    if K ==3:
        for i in range(N):    
            ac = xx[i]//I**2
            b = (xx[i]-(ac)*I**2)//I
            c = xx[i]%I
            ii[i,0] = ac
            ii[i,1] = b
            ii[i,2] = c

    if K==4:
        for i in range(N):
            ac = xx[i]//I**3
            b = (xx[i]-ac*I**3)//I**2
            c = (xx[i]-ac*I**3-b*I**2)//I
            d = xx[i]%I
            ii[i,0] = ac
            ii[i,1] = b
            ii[i,2] = c
            ii[i,3] = d
    for n in range(N):
        y[n] = mu +np.random.normal(sum([a[k][ii[n,k]] for k in range(K)]), 1)         
    for k in range(K):
        data[chr(97+k)] = ii[:,k].astype('int')
    data['y'] = y
    col_num =[ chr(97+k) for k in range(K)]
             
    if num == 3 and K==2:
        gamma = 1.2
        I_1 = np.ceil(np.power(S,rho)).astype('int')
        I_2 = np.ceil(np.power(S,cappa)).astype('int')
        P = np.random.uniform(np.power(S, 1-rho-cappa), gamma*np.power(S, 1-rho-cappa), size = (I_1,I_2) )
        P[P>1] = 1
        Z = np.random.binomial(n=1, p=P, size=(I_1, I_2))

        iss = [I_1, I_2]
        tau = np.ones(K)
        a = [np.random.normal(0,1/np.sqrt(tau[k]),size=iss[k]) for k in range(K)]  
        y= np.zeros(np.sum(Z))
        ii = np.zeros((np.sum(Z),K))
        aux = 0
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if Z[i,j] == 1:
                    y[aux] = np.random.normal(0 +a[0][i]+ a[1][j], 1)
                    ii[aux, 0] =i
                    ii[aux, 1] = j
                    aux += 1
                    
        col_num = ['a','b']
        data['a'] = ii[:,0].astype('int')
        data['b'] = ii[:,1].astype('int')
        data['y'] = y

    x.import_df(data, col_num) # these automatically takes care of eliminating 0
    if return_a:
        return x, np.array(a)
    else:
        return x


# generate the plots, given an input file containing the results of the simulations
def plot_results(file_name, K,tau_e, tau_k, delta, eps,reg_num,rand=True,vanilla = False, save= False,S=1, rho=1, cappa = 1, output_name =[], skip = False):
    data = pd.read_csv(file_name)
    I = data['I'].unique()
    bound = np.zeros((len(delta), len(I)))
    bound_vanilla = np.zeros((len(delta), len(I)))
    mixing_time = []
    
    aux1 = np.zeros(len(delta))
    aux2 = np.zeros(len(delta))
    for en,i in enumerate(I):

        print('#####################  '+ str(i)+ '  #################################################################\n' )
        if reg_num !=3:
            x= asymptotic_regimes(reg_num,K,I=i)
        else:
            x= asymptotic_regimes(3,K,I=i,S=S, rho=rho, cappa = cappa)
        # these automatically takes care of eliminating 0
        #rho[en], rho_coll[e], Sigma[e], B[en], B_coll[en]= x.conv_rate(tau_e,tau_k)
        rho, rho_coll, Sigma, B, B_coll= x.conv_rate(tau_e,tau_k)
        rho_sbsb_coll = np.max(np.abs(la.eigvals(Sigma[1:,1:] - B_coll@Sigma[1:,1:]@B_coll.transpose())))
        if vanilla:
             rho_sbsb_pv = np.max(np.abs(la.eigvals(Sigma- B@Sigma@B.transpose())))
        print(np.linalg.norm(B_coll))

        mixing_time.append(1/(1-rho_coll))
        a1=  (0.5*np.log(12+8*np.sqrt(2/np.pi))*erfinv(eps) +1/(2*np.sqrt(2)*np.e)) /(1-rho_coll)
        a2=  (0.5*np.mean(np.log( data[ (data['I']==I[en]) & (data['var']=='fixed') & (data['coll']=='collapsed')]['dist']   ))-0.5*np.log(rho_sbsb_coll) - np.log(2*np.sqrt(2)*erfinv(eps)))/(1-rho_coll)
        
        
        if K!=2:
            for di,d in enumerate(delta):

                n_d_coll=1
                aux = B_coll + 0
                if skip==False:
                    while ( (1- np.linalg.norm(aux)**(1/n_d_coll)) <= ((1-rho_coll)/(1+d))):
                        aux =aux@B_coll
                        n_d_coll+= 1
                else:
                    n_d_coll = 2

                bound[di, en]= (1+ np.max((n_d_coll,np.ceil(a1*(1+d)))))/(1-eps) + 1+ np.max((n_d_coll, np.ceil( a2*(1+d))))
                print( n_d_coll, np.ceil(a1*(1+d)), np.ceil( a2*(1+d)))
                if vanilla:
                    n_d=1
                    aux = B + 0
                    #while ( (1- np.linalg.norm(aux, '2')**(1/n_d)) <= ((1-rho)/(1+d))):
                    while ( (1- np.linalg.norm(aux)**(1/n_d)) <= ((1-rho)/(1+d))):
                        aux =aux@B
                        n_d+= 1

                    bound_vanilla[di, en]=(1+np.max((n_d, np.ceil((0.5 *np.log(12+8*np.sqrt(2/np.pi))*erfinv(eps)+ 1/(2*np.sqrt(2)*np.e))/(1-(rho))*(1+d)/(1-eps)))) +  1+  np.max((n_d, np.ceil((  (0.5*np.mean(np.log(data[ (data['I']==I[en]) & (data['var']=='fixed') & (data['coll']=='plain')]['dist']   ))-0.5*np.log(rho_sbsb_pv) - np.log(erfinv(eps)*2*np.sqrt(2))))/(1-rho)*(1+d)))) )

        else:
            d = 0
            n_d_coll = 0
            n_d = 0 
            bound[0, en]= 2+  np.ceil(a1)/(1-eps) +  np.ceil(a2)
            if vanilla:
                bound_vanilla[0, en]= 1+ np.ceil((0.5 *np.log(12+8*np.sqrt(2/np.pi))*erfinv(eps)+ 1/(2*np.sqrt(2)*np.e))/(1-(rho))*(1+d)/(1-eps)) +  1+  np.ceil(  (0.5*np.mean(np.log(data[ (data['I']==I[en]) & (data['var']=='fixed') & (data['coll']=='plain')]['dist']   ))-0.5*np.log(rho_sbsb_pv) - np.log(erfinv(eps)*2*np.sqrt(2))))/(1-rho)

        
    plt.figure(figsize=(10,5))
    font = {'fontname':'Cambria Math'}
    plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
    
    plt.xticks(range(len(I)), np.array(I)*K+1)
    if vanilla == False:
        for di,d in enumerate(delta):
            if K != 2:
                plt.plot(range(len(I)), bound[di,:],'--',label='bound, fixed var, delta='+str(d))
            else: 
                plt.plot(range(len(I)), bound[di,:],'--',label='bound, fixed var')
            plt.scatter(range(len(I)), bound[di,:],s= 200, marker = '*')
    for m in data[['coll','var']].drop_duplicates().iterrows():
        if vanilla == False:
            if m[1][0]!= "plain":
                plt.plot(range(len(I)),   [ 1+ np.mean(data.iloc[data.groupby(['coll','var','I']).groups[(str(m[1]['coll']),str(m[1]['var']),i)].tolist(),:]['t']) for i in I], label= (str(m[1]['coll'])+',  '+ str(m[1]['var'])))
                plt.scatter(range(len(I)),[ 1+ np.mean(data.iloc[data.groupby(['coll','var','I']).groups[(str(m[1]['coll']),str(m[1]['var']),i)].tolist(),:]['t']) for i in I], s=50)
        else:
            
            plt.plot(range(len(I)),[ 1+ np.mean(data.iloc[data.groupby(['coll','var','I']).groups[(str(m[1]['coll']),str(m[1]['var']),i)].tolist(),:]['t']) for i in I], label= (str(m[1]['coll'])+',  '+ str(m[1]['var'])))
            plt.scatter(range(len(I)),[ 1+ np.mean(data.iloc[data.groupby(['coll','var','I']).groups[(str(m[1]['coll']),str(m[1]['var']),i)].tolist(),:]['t']) for i in I], s=50)
    

    if vanilla:
        for di,d in enumerate(delta):
            if K!= 2:
                plt.plot(range(len(I)), bound_vanilla[di,:],'--',label='bound, fixed var')
            else:
                plt.plot(range(len(I)), bound_vanilla[di,:],'--',label='bound, fixed var')
            plt.scatter(range(len(I)), bound_vanilla[di,:],s= 200, marker = '*')
    
    

    plt.ylabel("Meeting time", fontsize=30  , **font)
    plt.xlabel("Parameters number", fontsize= 30, **font)
    plt.legend(fontsize=25)
    
    plt.rc('font',family='Cambria Math')
    matplotlib.rc('font',family='Cambria Math')
    plt.title("Average meeting times",fontsize = 30,**font)
    plt.ylim(0)
    plt.grid(True, which="both")
    if save:
        plt.savefig(str(output_name)+'.png', bbox_inches="tight")
    plt.show()


def run_experiment(K,I, J,tau_e, tau_k, eps, reg_num, rand=True,export_results = False,T_max = 200, filename=[], collapsed = [], variance = [], S=1, rho=1, cappa = 1):
    if len(collapsed) != len(variance):
        print("variants error")
        return -1
    
    distances = np.zeros((len(collapsed ), len(I), J))
    times = np.zeros((len(collapsed ), len(I), J))

    for en,i in enumerate(I):
        print('#####################  '+ str(i)+ '  #################################################################\n' )
        x = asymptotic_regimes(reg_num, K , i, S, rho, cappa)
        rho, rho_coll, Sigma , B, B_coll= x.conv_rate(tau_e,tau_k)
        rho_sbsb_pv = np.max(np.abs(la.eigvals(Sigma - B@Sigma@B.transpose())))
        rho_sbsb_coll = np.max(np.abs(la.eigvals(Sigma[1:,1:] - B_coll@Sigma[1:,1:]@B_coll.transpose())))
        epsi_coll = (erfinv(eps))**2*8/(np.linalg.norm(B_coll))**2*rho_sbsb_coll
        epsi_pv = (erfinv(eps))**2*8/(np.linalg.norm(B))**2*rho_sbsb_pv
        print(np.linalg.norm(B_coll),epsi_coll )
        print("go!")
        for e,(m,n) in enumerate(zip(collapsed,variance)):
            print("collapsed, fixed variance=", m,n)
            if m == False:
                epsi = epsi_pv
            else:
                epsi = epsi_coll
                print(epsi_coll)
            for j in range(J):
                a,b,distances[e,en,j], times[e,en,j]= MCMC_sampler_coupled(x, collapsed=m, PX=True, L=1, T=T_max, dist=epsi, rand= rand, var_fixed=n )
                print(times[e,en,j])
            pd.DataFrame(times.flatten()).to_csv("aux_"+str(K))    
                

    df = pd.DataFrame()
    df['t'] = times.flatten()
    df['dist'] = distances.flatten()
    df['I'] = np.hstack([np.repeat(I,J) for h in range(len(collapsed))])
    aux = [ [str(m)]*len(I)*J for m in collapsed]
    df['coll'] = [a for b in aux for a in b]
    df['coll'] = df['coll'].str.replace('True','collapsed')
    df['coll'] = df['coll'].str.replace('False','plain')
    aux = [ [str(n)]*len(I)*J for n in variance]
    df['var'] = [a for b in aux for a in b]
    df['var'] = df['var'].str.replace('True','fixed')
    df['var'] = df['var'].str.replace('False','free_v')
    

    if export_results:
        df.to_csv(filename)
        return -1
    else:
        return df
        

