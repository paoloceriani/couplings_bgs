import funct_simulation
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
norm = scipy.stats.norm(0,1)
import pandas as pd
from time import sleep
import random
import pandas as pd
from tqdm import tqdm

def countmemb(itr,I):
	count = np.zeros((itr.shape[1],I))
	for d in range(itr.shape[1]):
	    for val in itr[:,d]:
	        count[d,val] += 1
	return count

def distance(val_1,val_2):
   return  (val_1.tau-val_2.tau)@(val_1.tau-val_2.tau) + (val_1.mu-val_2.mu)**2+ np.sum([(val_1.a[cappa] - val_2.a[cappa])**2 for cappa in range(len(val_1.a))])
   
def convert_col(x):
    return x.replace({i:e for e,i in enumerate(set(x))})

def sum_on_all_new(a, data,k):
    #limitiamo il numero di liste 
    K = len(a)
    res = np.zeros(data.iss[k])
    kr = np.array(range(data.K))
    for l in kr[kr!= k]:
        res += data.N_sl[k][l]@a[l]
    return res

# Data class for Model 2 with laplace response
class Data_lapl:
    def __init__(self):
        self.K : int= None
        self.I: int = None
        self.N: int = None
        self.mu:float = None
        self.ii: numpy.ndarray = None
        self.iss: numpy.ndarray = None
        self.mean_y: numpy.float64 = None
        self.mean_y_s:numpy.ndarray = None
        self.N_s: numpy.ndarray = None
        self.N_sl : list= None 
        self.tau_k: numpy.ndarray = None
        self.b = None
    
    def generate(self,N,K,I,mu, tau_k,b):
        self.K = K
        self.I = I
        self.N = N
        self.b = b
        self.tau_k = tau_k  
        self.ii = np.random.randint(0,self.I, size=(self.N,self.K))
        self.iss = np.max(self.ii, axis = 0)+1 
        self.mu = mu
        self.a = np.random.normal(0,1/np.sqrt(tau_k), size = (self.I,self.K))
        aux = np.sum(self.a[self.ii,np.arange(self.K)], axis = 1) 
        self.y = np.random.laplace(aux + self.mu, b,size = self.N)
        self.mean_y = np.mean(self.y)
        self.mean_y_s = np.array([np.array([np.mean(self.y[self.ii[:,k] == h]) if not np.isnan(np.mean(self.y[self.ii[:,k] == h])) else 0 for h in range(0,self.iss[k])]) for k in range(self.K) ])
        self.N_s = countmemb(self.ii,self.I)
        self.N_sl = [[ np.zeros((self.iss[i],self.iss[j])) for j in range(self.K)] for i in range(self.K)]
        for i in range(self.K) :
            for j in range(self.K):
                for n in range(self.N):
                    self.N_sl[i][j][self.ii[n,i], self.ii[n,j]] += 1
    
    def import_df(self,df, col_name, mu, tau_k = np.ones(1),b=1):
        
        self.tau_k = tau_k+0 # se lo passo così sono a posto
        self.y = np.array(df['y'])
        self.ii= np.array(df.loc[:, col_name].apply(convert_col, axis = 0)).astype('int')
        self.I = self.ii.max()+1
        self.K = self.ii.shape[1]
        self.N= self.ii.shape[0]
        self.mu = mu
        self.b = b
        self.iss = np.max(self.ii, axis = 0).astype('int')+1 # it is the number of categories and the maximum 
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
        

    
# iteration value for Model 2 with laplace response 
class iter_value_laplace:  
    l = 2.38 
    def __init__(self, data,T,a, mu_pr = 0, tau_pr = 0, rand = True,  aux_to_save = 1,b=1):
        self.mu_pr = mu_pr
        self.tau_pr = tau_pr
        if rand == False:
            self.tau: numpy.ndarray = data. tau_k+0
            self.mu: float  =  np.random.normal(mu_pr,1)
            self.a: list = [np.random.normal(0,1, size = (data.iss[k])) for k in range(data.K)]
            
        else: 
            self.tau: numpy.ndarray = np.random.gamma(1,2,size = data.K)
            self.mu: float =  np.random.normal(mu_pr,0.3)
            self.a: list = [np.random.normal(0,2, size = (data.iss[k])) for k in range(data.K)] #otherwise variances explodes...
        self.a_means: numpy.ndarray = np.array([self.a[k]@data.N_s[k]/data.N for k in range(data.K)])
        self.mu_chain = np.zeros(1+T*data.K)
        self.a_means_chain = np.zeros((data.K,T+1 ))
        self.tau_chain = np.zeros((data.K,T+1 ))
        self.mu_chain[0]= self.mu
        self.b = b
        self.tau_chain[:,0] = self.tau
        self.a_means_chain[:,0]= self.a_means
        self.a_aux =  np.zeros(T+1)
        self.list_saving = []
        self.a_aux =  np.zeros((aux_to_save, T+1))
        for j in range(aux_to_save):
            k = np.random.randint(0,data.K)
            i = np.random.randint(0,data.I)
            self.list_saving.append((k,i))
            self.a_aux[j,0] = a[k][i] 

    # update for the marginal chain
    def update(self, data, t, mask,S, fixed =False, save_aux =False, proposal_std= -1):
        # we do local centering and update of the mu's    
        for k in range(data.K):
            csi_k = self.a[k] + self.mu
            self.mu = np.random.normal((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]))
            # old values 
            old_all = [np.array(data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(data.K)], axis = 0))+  self.a[k][i] for i in range(len(csi_k))]
            
            for s in range(S):
                aux_m = [old_all[i]-  csi_k[i] for i in range(len(csi_k))]
                z = np.random.normal(0, 1, size = len(csi_k))
                x = csi_k + z*proposal_std*np.sqrt(2)
                aux_mt = [old_all[i] -x[i] for i in range(len(csi_k))]
                
                for i in range(len(csi_k)):
                    acc_1 = - (np.sum(np.abs(aux_mt[i]))-np.sum(np.abs(aux_m[i])))/self.b
                    acc_3 = -self.tau[k]/2* ((x[i]-self.mu)**2-(csi_k[i] - self.mu)**2)

                    log_acc = acc_1 + acc_3
                    
                    if np.log(np.random.rand())< log_acc: 
                        csi_k[i] = x[i]
                        #aux[i]+=1

            self.a[k] = csi_k - self.mu
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            if fixed == False:
                self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
            self.mu_chain[1+t*data.K+k]=self.mu
        if save_aux == True:
            for en,(k,i) in enumerate(self.list_saving):
                self.a_aux[en,t+1] = self.a[k][i]
        
        self.tau_chain[:, t+1] = self.tau
        self.a_means_chain[:, t+1] = self.a_means

        
    # coupled update of local centering algorithm with laplace response
    def coupled_update(self, val_2, data,t,l,mask,b,S, fixed = False, close = False, factorized_proposals = False, factorized_acceptance = True, a_pvb= True, a_pvf = True):        
        for k in range(data.K):
            csi_k = self.a[k] + self.mu
            csi_k_2 = val_2.a[k] + val_2.mu
            
            mu_mean =   (self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k])
            mu_mean_2 = (self.tau_pr*self.mu_pr + val_2.tau[k]*np.sum(csi_k_2))/(self.tau_pr+val_2.tau[k]*data.iss[k])
            
            std_1 = 1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k])
            std_2 = 1/np.sqrt(self.tau_pr+val_2.tau[k]*data.iss[k])
            
            if close:
                self.mu, val_2.mu = funct_simulation.maximal_coupling_normal_1d(mu_mean, mu_mean_2, std_1, std_2)
            else:
                aux = np.random.normal(0,1)
                self.mu = mu_mean + aux*std_1
                val_2.mu = mu_mean_2 + aux*std_2
                
             
            old_all =  ([np.array(data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(data.K)], axis = 0))+  self.a[k][i] for i in range(len(csi_k))])
            old_all_2= ([np.array(data.y[mask[k][i]] - np.sum([val_2.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(data.K)], axis = 0))+  val_2.a[k][i] for i in range(len(csi_k))])# dimensione corretta

            
            x=y= np.zeros(len(csi_k))
            if a_pvb:
                proposal_var_b = 1/(data.N_s[k]*(1/(2*b**2))+self.tau[k])*2 
            else:
                proposal_var_b = 1/(data.N_s[k]*(1/(2*b**2))+self.tau[k])*l**2/len(self.a[k])
            
	    for s in range(S):
                aux_m = [old_all[i]-  csi_k[i] for i in range(len(csi_k))]
                aux_m2 = [old_all_2[i] -  csi_k_2[i] for i in range(len(csi_k))]  # questo cambia ad ogni update
                
                if factorized_proposals == False:
                	  
                    if close:
                        x,y= funct_simulation.reflection_coupling(csi_k,csi_k_2, np.identity(len(csi_k))*proposal_var_b, diag = True)
                    else:
                        z = np.random.normal(0,1, size = len(csi_k))
                        x = csi_k + z*proposal_var_b
                        y = csi_k_2 +z*proposal_var_b
                else:
                    if a_pvf:
                        proposal_var_f = 1/ (data.N_s[k]*(1/(2*b**2))+self.tau[k])*2
                    else:
                        proposal_var_f = 2*np.ones(len(self.a[k]))
                        
                    for i in range(len(csi_k)):
                        if close:
                            x[i],y[i]= funct_simulation.maximal_coupling_normal_1d(csi_k[i],csi_k_2[i], np.sqrt(proposal_var_f[i]),np.sqrt(proposal_var_f[i]))
                        else:
                            z = np.random.normal(0,np.sqrt(proposal_var_f[i]))
                            x[i] = csi_k[i] + z
                            y[i] = csi_k_2[i] +z
                aux_mt = [old_all[ii] -x[ii] for ii in range(len(csi_k))]
                aux_mt2 =[old_all_2[ii] -y[ii] for ii in range(len(csi_k))]   
               
                if factorized_acceptance:
                    for i in range(len(csi_k)):
                        acc_1 = - (np.sum(np.abs(aux_mt[i]))-np.sum(np.abs(aux_m[i])))/self.b
                        acc_3 = - self.tau[k]/2*((x[i]-self.mu)**2-(csi_k[i] - self.mu)**2)
                        acc_12 = -(np.sum(np.abs(aux_mt2[i]))-np.sum(np.abs(aux_m2[i])))/self.b
                        acc_32 = -val_2.tau[k]/2*((y[i]-val_2.mu)**2-(csi_k_2[i] - val_2.mu)**2)
                        
                        log_acc = acc_1 + acc_3
                        log_acc2 = acc_12 + acc_32
                        u = np.log(np.random.rand())
                        if u < log_acc: 
                            csi_k[i] = x[i]
                        if u < log_acc2: 
                            csi_k_2[i] = y[i]

                else:
                    acc_1 = acc_3 = acc_12 = acc_32 =  0
                    for i in range(len(csi_k)):
                        acc_1+= - (np.sum(np.abs(aux_mt[i]))-np.sum(np.abs(aux_m[i])))/self.b
                        acc_3+= - self.tau[k]/2*((x[i]-self.mu)**2-(csi_k[i] - self.mu)**2)

                        acc_12+= -(np.sum(np.abs(aux_mt2[i]))-np.sum(np.abs(aux_m2[i])))/self.b
                        acc_32+= -val_2.tau[k]/2*((y[i]-val_2.mu)**2-(csi_k_2[i] - val_2.mu)**2)
                    log_acc = acc_1 + acc_3
                    log_acc2 = acc_12 + acc_32
                    u = np.log(np.random.rand())
                    if u < log_acc:
                        csi_k = x
                    if u < log_acc2:
                        csi_k_2 = y
                        

            self.a[k] = csi_k - self.mu
            val_2.a[k] = csi_k_2 - val_2.mu
            
            self.a_means[k] =  self.a[k]@data.N_s[k]/data.N
            val_2.a_means[k] =  val_2.a[k]@data.N_s[k]/data.N
            
            if fixed==False:
                self.tau[k], val_2.tau[k] = funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5,  2/np.sum(np.power(self.a[k],2)),  2/np.sum(np.power(val_2.a[k],2)))
     
            self.mu_chain[t*data.K+1+k]=self.mu
            val_2.mu_chain[t*data.K+1+k]=val_2.mu
            
        self.tau_chain[:, t+1] = self.tau
        self.a_means_chain[:, t+1] = self.a_means
        
        val_2.tau_chain[:, t+1] = val_2.tau
        val_2.a_means_chain[:, t+1] = val_2.a_means

        if (self.mu == val_2.mu) and np.array([(self.a[k] == val_2.a[k]).all() for k in range(data.K)]).all() and (self.tau == val_2.tau).all():
            print("yeeeee")           
            return -1
    def distance(self,val_2):
	    return (self.mu-val_2.mu)**2 +  sum([(self.a[k]-val_2.a[k])@(self.a[k]-val_2.a[k]) for k in range(len(self.a))]) + (self.tau -val_2.tau)@(self.tau -val_2.tau)

    
def asymptotic_regimes_lapl(num,K,I=10,b=1,mu=0, tau_k = np.ones(1), return_a = False):
    x =Data_lapl()
    data = pd.DataFrame()
    if num ==1:
        # dense regime
        p = 0.1/(I**(K-2))
    elif num == 2:
        # sparse 
        p = 10/(I**(K-1))
    elif num == 4:
        # half
        p = 0.1/(I**(K-1.5))
    
    N = np.random.binomial(I**K, p ,1)[0]
    iss = [I]*K
    tau = np.ones(K)
    a = np.array([np.random.normal(0,1/np.sqrt(tau_k[k]),size=iss[k]) for k in range(K)])
    y= np.zeros(np.sum(N))
    ii = np.zeros((N,K)).astype(int)
        
    xx = random.sample(range(I**K),N) 
    if K == 2:
        for i in range(N):
            ac = xx[i]//I
            bi = xx[i]-ac*I
            ii[i,0] = ac
            ii[i,1] = bi
    if K ==3:
        for i in range(N):    
            ac = xx[i]//I**2
            bi = (xx[i]-(ac)*I**2)//I
            c = xx[i]%I
            ii[i,0] = ac
            ii[i,1] = bi
            ii[i,2] = c

    if K==4:
        for i in range(N):
            ac = xx[i]//I**3
            bi = (xx[i]-ac*I**3)//I**2
            c = (xx[i]-ac*I**3-bi*I**2)//I
            d = xx[i]%I
            ii[i,0] = ac
            ii[i,1] = bi
            ii[i,2] = c
            ii[i,3] = c
    for n in range(N):
        y[n] = np.random.laplace(sum([a[k][ii[n,k]] for k in range(K)])+ mu, b)         
    for k in range(K):
        data[chr(97+k)] = ii[:,k].astype('int')
    data['y'] = y
    col_num =[ chr(97+k) for k in range(K)]

    x.import_df(data, col_num,mu, tau_k,b) # these automatically takes care of eliminating 0
    x.a = a
    if return_a:
        return x,a
    else:
        return x
    
