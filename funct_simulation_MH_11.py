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
	# in this way itr might have different dimensions for different factor, no good
	count = np.zeros((itr.shape[1],I))
	for d in range(itr.shape[1]):
	    for val in itr[:,d]:
	        count[d,val] += 1
	return count

def distance(val_1,val_2):
# compute square distances between vector of ALL parameters    
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

#initializing the values for the chain, and update them
class iter_value_g2:  
    l = 2.38 #value for sampling
    def __init__(self, data,T, mu_pr = 0, tau_pr = 1, rand = True):
        self.mu_pr = mu_pr
        self.tau_pr = tau_pr
        
        if rand == False:
            self.tau = np.ones(data.K)
            self.tau_e = 1
            #self.mu =  mu+0
            #self.a = [data.a[:,k]+0 for k in range(data.K)]
            self.mu =  np.random.normal(mu_pr,0.3)
            self.a = [np.random.normal(0,3, size = (data.iss[k])) for k in range(data.K)]
            
        else: 
            self.tau_e = np.random.gamma(1,2)
            self.tau = np.random.gamma(1,1,size = data.K)
            self.mu =  np.random.normal(mu_pr,0.3)
            self.a = [np.random.normal(0,3, size = (data.iss[k])) for k in range(data.K)] #otherwise variances explodes...
        #self.a = [np.random.normal(0,2, size = (data.iss[k])) for k in range(data.K)] #otherwise variances explodes...
        self.a_means = np.array([self.a[k]@data.N_s[k]/data.N for k in range(data.K)])
        
        self.mu_chain = np.zeros(1+T*data.K)
        self.a_means_chain = np.zeros((data.K,T+1 ))
        self.tau_chain = np.zeros((data.K,T+1 ))
        self.tau_e_chain = np.zeros(T+1)
        self.mu_chain[0]= self.mu
        self.tau_chain[:,0] = self.tau
        self.tau_e_chain[0] = self.tau_e
        self.a_means_chain[:,0]= self.a_means
        self.a_aux =  np.zeros(T+1)
        
    
          
    def update(self, data, t, fixed =False):
        # we do local centering and update of the mu's
        pred = np.zeros(data.N)
        for k in range(data.K):
            # do centering
            csi_k = self.a[k] + self.mu
           
            # update mu
            self.mu = np.random.normal((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]))
            sum_a_l = sum_on_all_new(self.a, data,k)
            mean_csi = (data.N_s[k]*data.mean_y_s[k]- sum_a_l) *self.tau_e/(self.tau[k]+ data.N_s[k]*self.tau_e) + self.mu*self.tau[k]/(self.tau[k]+ data.N_s[k]*self.tau_e)
            var_csi = 1/(self.tau[k]+ data.N_s[k]*self.tau_e)
            for i in range(len(csi_k)):
                # old= data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(K)], axis = 0)+  self.a[k][i] # dimensione corretta    
                csi_k[i] = np.random.normal( mean_csi[i] , np.sqrt(var_csi[i]))

            self.a[k] = csi_k - self.mu

            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            self.mu_chain[1+t*data.K+k]=self.mu
            pred += self.a[k][data.ii[:,k]]
            
            if fixed == False:
                self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
            
        pred += self.mu
        if fixed == False:
            SS0 = np.sum(np.power((data.y-pred),2))
            self.tau_e =  np.random.gamma(size=1, shape= data.N/2-0.5, scale=2/SS0)
        self.tau_chain[:, t+1] = self.tau
        self.tau_e_chain[t+1] = self.tau_e
        self.a_means_chain[:, t+1] = self.a_means
        
        
    def coupled_update(self,val_2, data, t,l, fixed =False, close= False):
        # we do local centering and update of the mu's
        pred = np.zeros(data.N)
        pred2 = np.zeros(data.N)
        for k in range(data.K):
            # do centering
            csi_k = self.a[k] + self.mu
            csi_k2 = val_2.a[k] + val_2.mu

            self.mu, val_2.mu = funct_simulation.maximal_coupling_normal_1d((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),(self.tau_pr*self.mu_pr + val_2.tau[k]*np.sum(csi_k2))/(self.tau_pr+val_2.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]), 1/np.sqrt(self.tau_pr+val_2.tau[k]*data.iss[k])  )
            
            sum_a_l = sum_on_all_new(self.a, data,k)
            sum_a_l2 = sum_on_all_new(val_2.a, data,k)

            mean_csi = (data.N_s[k]*data.mean_y_s[k]- sum_a_l) *self.tau_e  /(self.tau[k]+ data.N_s[k]*self.tau_e)   + self.mu*self.tau[k]/(self.tau[k]+ data.N_s[k]*self.tau_e)
            mean_csi2 =(data.N_s[k]*data.mean_y_s[k]- sum_a_l2)*val_2.tau_e/(val_2.tau[k]+ data.N_s[k]*val_2.tau_e) + val_2.mu*val_2.tau[k]/(val_2.tau[k]+ data.N_s[k]*val_2.tau_e)
            
            if close:
                if ((self.tau[k]== val_2.tau[k]) and (val_2.tau_e == self.tau_e)):
                    csi_k,csi_k2= funct_simulation.reflection_coupling(mean_csi,mean_csi2, np.identity(len(csi_k))/(self.tau[k]+ data.N_s[k]*self.tau_e), diag=True)
                else:
                    csi_k,csi_k2= funct_simulation.maximal_coupling_normal_d(mean_csi,mean_csi2, np.identity(len(csi_k))/(self.tau[k]+ data.N_s[k]*self.tau_e), np.identity(len(csi_k))/(val_2.tau[k]+ data.N_s[k]*val_2.tau_e), diag = True)
            else:
                z = np.random.normal(0,1, size = len(csi_k))
                csi_k = mean_csi +  z/(self.tau[k]+ data.N_s[k]*self.tau_e)
                csi_k2 = mean_csi2 +  z/(val_2.tau[k]+ data.N_s[k]*val_2.tau_e)
                
            self.a[k] = csi_k - self.mu
            val_2.a[k] = csi_k2 - val_2.mu
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            val_2.a_means[k] = val_2.a[k]@data.N_s[k]/data.N
            self.mu_chain[1+t*data.K+k]=self.mu
            val_2.mu_chain[1+t*data.K+k]=val_2.mu
            pred += self.a[k][data.ii[:,k]]
            pred2 += val_2.a[k][data.ii[:,k]]
            if np.all(self.a[k]==0) or np.all(val_2.a[k]==0):
                print(self.a[k], val_2.a[k], self.tau[k], val_2.tau[k],self.mu, val_2.mu, t, k)
                
            if fixed == False:
                self.tau[k], val_2.tau[k]= funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5, 2/np.sum(np.power(self.a[k],2)), 2/np.sum(np.power(val_2.a[k],2))) 
              
                #if close:
                #    self.tau[k], val_2.tau[k]= funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5, 2/np.sum(np.power(self.a[k],2)), 2/np.sum(np.power(val_2.a[k],2))) 
                #else:
                #    u = np.random.randint(500)
                #    np.random.seed(u)
                #    self.tau_e = np.random.gamma(shape = data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
                #    np.random.seed(u)
                #    val_2.tau_e = np.random.gamma(shape = data.iss[k]/2-0.5, scale= 2/np.sum(np.power(val_2.a[k],2)))

        pred += self.mu
        pred2 += val_2.mu
        if fixed == False:
            SS0 = np.sum(np.power((data.y-pred),2))
            SS02 = np.sum(np.power((data.y-pred2),2))
            self.tau_e, val_2.tau_e = funct_simulation.maximal_coupling_gamma(data.N/2-0.5,2/SS0,2/SS02 )
            
        self.tau_chain[:, t+1] = self.tau
        self.tau_e_chain[t+1] = self.tau_e
        self.a_means_chain[:, t+1] = self.a_means
        
        val_2.tau_chain[:, t+1] = val_2.tau
        val_2.tau_e_chain[t+1] = val_2.tau_e
        val_2.a_means_chain[:, t+1] = val_2.a_means
        if (self.mu == val_2.mu) and np.array([(self.a[k] == val_2.a[k]).all() for k in range(data.K)]).all() and (self.tau == val_2.tau).all() and self.tau_e == val_2.tau_e:
            print("yeeeee")           
            return -1
        
        
        
    def coupled_update_MH(self, val_2, data,t,l, S=1, fixed = False, close = False):

        # we do local centering and update of the mu's
        pred = np.zeros(data.N)
        pred2 = np.zeros(data.N)
        for k in range(data.K):
            # do centering
            csi_k = self.a[k] + self.mu
            csi_k2 = val_2.a[k] + val_2.mu
           
            # update mu  
            self.mu, val_2.mu = funct_simulation.maximal_coupling_normal_1d((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),(self.tau_pr*self.mu_pr + val_2.tau[k]*np.sum(csi_k2))/(self.tau_pr+val_2.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]), 1/np.sqrt(self.tau_pr+val_2.tau[k]*data.iss[k])  )
            
            sum_a_l = sum_on_all_new(self.a, data,k)
            sum_a_l2= sum_on_all_new(val_2.a, data,k)
                      
            c_ik =  1/(self.tau[k] + self.tau_e*data.N_s[k])
            c_ikt = 1/(self.tau[k] + self.tau_e*data.N_s[k])  
            
            c_ik2 =  1/(val_2.tau[k] + val_2.tau_e*data.N_s[k])
            c_ikt2 = 1/(val_2.tau[k] + val_2.tau_e*data.N_s[k])    

            m_ik =  ((data.N_s[k]*data.mean_y_s[k] - sum_a_l)*self.tau_e+ self.mu*self.tau[k])*c_ik
            m_ikt = ((data.N_s[k]*data.mean_y_s[k] - sum_a_l)*self.tau_e+ self.mu*self.tau[k])*c_ikt
            
            m_ik2 =  ((data.N_s[k]*data.mean_y_s[k] - sum_a_l2)*val_2.tau_e+ val_2.mu*val_2.tau[k])*c_ik2
            m_ikt2 = ((data.N_s[k]*data.mean_y_s[k] - sum_a_l2)*val_2.tau_e+ val_2.mu*val_2.tau[k])*c_ikt2

            
            
            for s in range(S):
                if close:
                    if ((self.tau[k]== val_2.tau[k]) and (val_2.tau_e == self.tau_e)):
                        x,x2= funct_simulation.reflection_coupling(m_ik,m_ik2, np.identity(len(csi_k))/(self.tau[k]+ data.N_s[k]*self.tau_e), diag=True)
                    else:
                        x,x2= funct_simulation.maximal_coupling_normal_d(m_ik,m_ik2, np.identity(len(csi_k))/(self.tau[k]+ data.N_s[k]*self.tau_e), np.identity(len(csi_k))/(val_2.tau[k]+ data.N_s[k]*val_2.tau_e), diag = True)
           
                else:
                    z = np.random.normal(0,1, size = len(csi_k))
                    x = m_ik + z*c_ik
                    x2 = m_ik2 +z*c_ik2
                        
                
                for i in range(len(csi_k)):
                    acc_1 = - self.tau_e/2*(csi_k[i]-x[i])*(2*(data.N_s[k][i]*data.mean_y_s[k][i] - sum_a_l[i])-(x[i]+csi_k[i])*data.N_s[k][i] )
                    acc_2 = - self.tau[k]/2*((x[i]-csi_k[i])*(x[i]+csi_k[i]-2*self.mu))
                    acc_3 = - ((csi_k[i] + x[i] - m_ik[i] - m_ikt[i])*(csi_k[i] - x[i] + m_ik[i] - m_ikt[i] ))/(2*c_ik[i])
                    
                    log_acc = acc_1+acc_2 + acc_3
                    
                    u = np.random.rand()
                    if np.log(u)< log_acc: 
                        csi_k[i] = x[i]
                            
                    acc_12 = - val_2.tau_e/2*(csi_k2[i]-x2[i])*(2*(data.N_s[k][i]*data.mean_y_s[k][i] - sum_a_l2[i])-(x2[i]+csi_k2[i])*data.N_s[k][i] )
                    acc_22 = - val_2.tau[k]/2*((x2[i]-csi_k2[i])*(x2[i]+csi_k2[i]-2*val_2.mu))
                    acc_32 = - ((csi_k2[i] + x2[i] - m_ik2[i] - m_ikt2[i])*(csi_k2[i] - x2[i] + m_ik2[i] - m_ikt2[i] ))/(2*c_ik2[i])
                     
                    log_acc2 = acc_12+acc_22 + acc_32

                    if np.log(u)< log_acc2: 
                        csi_k2[i] = x2[i]
                      

                
            self.a[k] = csi_k - self.mu
            val_2.a[k] = csi_k2 - val_2.mu
            
            if np.all(self.a[k]==0) or np.all(val_2.a[k]==0):
                print(self.a[k], val_2.a[k], self.tau[k], val_2.tau[k],self.mu, val_2.mu, t, k)
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            val_2.a_means[k] = val_2.a[k]@data.N_s[k]/data.N
            
            self.mu_chain[1+t*data.K+k]=self.mu
            val_2.mu_chain[1+t*data.K+k]=val_2.mu
            pred += self.a[k][data.ii[:,k]]
            pred2 += val_2.a[k][data.ii[:,k]]
            
            if fixed == False:
                self.tau[k], val_2.tau[k]= funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5, 2/np.sum(np.power(self.a[k],2)), 2/np.sum(np.power(val_2.a[k],2))) 
                #if close:
                #    self.tau[k], val_2.tau[k]= funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5, 2/np.sum(np.power(self.a[k],2)), 2/np.sum(np.power(val_2.a[k],2))) 
                #else:
                #    u = np.random.randint(500)
                #    np.random.seed(u)
                #    self.tau_e = np.random.gamma(shape = data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
                #    np.random.seed(u)
                #    val_2.tau_e = np.random.gamma(shape = data.iss[k]/2-0.5, scale= 2/np.sum(np.power(val_2.a[k],2)))

        pred += self.mu
        pred2 += val_2.mu
        if fixed == False:
            SS0 = np.sum(np.power((data.y-pred),2))
            SS02 = np.sum(np.power((data.y-pred2),2))
            self.tau_e, val_2.tau_e = funct_simulation.maximal_coupling_gamma(data.N/2-0.5,2/SS0,2/SS02 )
            
        self.tau_chain[:, t+1] = self.tau
        self.tau_e_chain[t+1] = self.tau_e
        self.a_means_chain[:, t+1] = self.a_means
        
        val_2.tau_chain[:, t+1] = val_2.tau
        val_2.tau_e_chain[t+1] = val_2.tau_e
        val_2.a_means_chain[:, t+1] = val_2.a_means
        
        
        
        if (self.mu == val_2.mu) and np.array([(self.a[k] == val_2.a[k]).all() for k in range(data.K)]).all() and (self.tau == val_2.tau).all() and self.tau_e == val_2.tau_e:
            print("yeeeee")           
            return -1

    def coupled_update_MH_RW(self, val_2, data,t,l, S=1, fixed = False, close = False, factorized = False):

        pred = np.zeros(data.N)
        pred2 = np.zeros(data.N)
        for k in range(data.K):
            csi_k = self.a[k] + self.mu
            csi_k2 = val_2.a[k] + val_2.mu

            self.mu, val_2.mu = funct_simulation.maximal_coupling_normal_1d((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),(self.tau_pr*self.mu_pr + val_2.tau[k]*np.sum(csi_k2))/(self.tau_pr+val_2.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]), 1/np.sqrt(self.tau_pr+val_2.tau[k]*data.iss[k])  )
            
            sum_a_l = sum_on_all_new(self.a, data,k)
            sum_a_l2= sum_on_all_new(val_2.a, data,k)

            for s in range(S):
            
                c_ik =  1/(self.tau[k] + self.tau_e*data.N_s[k]) 
                x = np.zeros(len(csi_k))
                x2 = np.zeros(len(csi_k2))
                if factorized == False:
                    if close:
                        x,x2= funct_simulation.reflection_coupling(csi_k,csi_k2, np.identity(len(csi_k))*np.sqrt(c_ik*2), diag=True)
                    else:
                        z = np.random.normal(0,1, size = len(csi_k))
                        x = csi_k + z*c_ik*np.sqrt(2)
                        x2 = csi_k2 +z*c_ik*np.sqrt(2)

                for i in range(len(csi_k)):
                    if factorized == True:
                        if close:
                            x[i],x2[i] = funct_simulation.reflection_coupling_1d(csi_k[i],csi_k2[i],np.sqrt(c_ik[i]*2))
                        else:
                            z = np.random.normal(0,1)
                            x[i] = csi_k[i] + z*c_ik[i]*np.sqrt(2)
                            x2[i] = csi_k2[i] +z*c_ik[i]*np.sqrt(2)

                        
                    acc_1 = - self.tau_e/2*(csi_k[i]-x[i])*(2*(data.N_s[k][i]*data.mean_y_s[k][i] - sum_a_l[i])-(x[i]+csi_k[i])*data.N_s[k][i] )
                    acc_2 = - self.tau[k]/2*((x[i]-csi_k[i])*(x[i]+csi_k[i]-2*self.mu))

                    log_acc = acc_1+acc_2 
                    
                    u = np.random.rand()
                    if np.log(u)< log_acc: 
                        csi_k[i] = x[i]
                            
                    acc_12 = - val_2.tau_e/2*(csi_k2[i]-x2[i])*(2*(data.N_s[k][i]*data.mean_y_s[k][i] - sum_a_l2[i])-(x2[i]+csi_k2[i])*data.N_s[k][i] )
                    acc_22 = - val_2.tau[k]/2*((x2[i]-csi_k2[i])*(x2[i]+csi_k2[i]-2*val_2.mu))
                    
                    log_acc2 = acc_12+acc_22

                    if np.log(u)< log_acc2: 
                        csi_k2[i] = x2[i]

            self.a[k] = csi_k - self.mu
            val_2.a[k] = csi_k2 - val_2.mu
            
            if np.all(self.a[k]==0) or np.all(val_2.a[k]==0):
                print(self.a[k], val_2.a[k], self.tau[k], val_2.tau[k],self.mu, val_2.mu, t, k)
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            val_2.a_means[k] = val_2.a[k]@data.N_s[k]/data.N
            
            self.mu_chain[1+t*data.K+k]=self.mu
            val_2.mu_chain[1+t*data.K+k]=val_2.mu
            pred += self.a[k][data.ii[:,k]]
            pred2 += val_2.a[k][data.ii[:,k]]
            
            if fixed == False:
                self.tau[k], val_2.tau[k]= funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5, 2/np.sum(np.power(self.a[k],2)), 2/np.sum(np.power(val_2.a[k],2))) 
                pred += self.mu
        pred2 += val_2.mu
        if fixed == False:
            SS0 = np.sum(np.power((data.y-pred),2))
            SS02 = np.sum(np.power((data.y-pred2),2))
            self.tau_e, val_2.tau_e = funct_simulation.maximal_coupling_gamma(data.N/2-0.5,2/SS0,2/SS02 )
            
        self.tau_chain[:, t+1] = self.tau
        self.tau_e_chain[t+1] = self.tau_e
        self.a_means_chain[:, t+1] = self.a_means
        
        val_2.tau_chain[:, t+1] = val_2.tau
        val_2.tau_e_chain[t+1] = val_2.tau_e
        val_2.a_means_chain[:, t+1] = val_2.a_means
        

        if (self.mu == val_2.mu) and np.array([(self.a[k] == val_2.a[k]).all() for k in range(data.K)]).all() and (self.tau == val_2.tau).all() and self.tau_e == val_2.tau_e:
            print("yeeeee")           
            return -1

        
    def update_MH(self, data, t,S=1,  fixed =False):
        
        # we do local centering and update of the mu's
        pred = np.zeros(data.N)
        for k in range(data.K):
            # do centering
            csi_k = self.a[k] + self.mu
           
            # update mu
            self.mu = np.random.normal((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]))
            sum_a_l = sum_on_all_new(self.a, data,k)
            
            csi_k = self.a[k] + self.mu              
            c_ik =  1/(self.tau[k] + self.tau_e*data.N_s[k])
            c_ikt = 1/(self.tau[k] + self.tau_e*data.N_s[k])         
            
            m_ik =  ((data.N_s[k]*data.mean_y_s[k] - sum_a_l)*self.tau_e+ self.mu*self.tau[k])*c_ik
            m_ikt = ((data.N_s[k]*data.mean_y_s[k] - sum_a_l)*self.tau_e+ self.mu*self.tau[k])*c_ikt

            for s in range(S):
                x = np.random.normal(m_ik, np.sqrt(c_ik)) 
                for i in range(len(csi_k)):
                    auxi = 0
                    # do some steps of metropolis
                    #acc_1 = - self.tau_e/2*(np.sum((old -x)**2)-np.sum((old -  csi_k[i])**2)) 
                    acc_1 = - self.tau_e/2*(csi_k[i]-x[i])*(2*(data.N_s[k][i]*data.mean_y_s[k][i] - sum_a_l[i])-(x[i]+csi_k[i])*data.N_s[k][i] )
                    acc_2 = - self.tau[k]/2*(x[i]-csi_k[i])*(x[i]+csi_k[i]-2*self.mu)
                    acc_3 = - ((csi_k[i] + x[i] - m_ik[i] - m_ikt[i])*(csi_k[i] - x[i] + m_ik[i] - m_ikt[i] ))/(2*c_ik[i])
                    
                    log_acc = acc_1+acc_2 + acc_3

                    if np.log(np.random.rand())< log_acc: 
                        csi_k[i] = x[i]
              #          auxi +=1
                        
             #   acc_prob[k,i] = ((t) * acc_prob[k,i] + auxi /S)/(t+1)   
                
                
            self.a[k] = csi_k - self.mu

            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            if fixed == False:
                self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
            self.mu_chain[1+t*data.K+k]=self.mu
            pred += self.a[k][data.ii[:,k]]
        pred += self.mu
        
        
        if fixed == False:
            SS0 = np.sum(np.power((data.y-pred),2))
            self.tau_e =  np.random.gamma(size=1, shape= data.N/2-0.5, scale=2/SS0)
        self.tau_chain[:, t+1] = self.tau
        self.tau_e_chain[t+1] = self.tau_e
        self.a_means_chain[:, t+1] = self.a_means
        
        
        
        
    def update_MH_RW(self, data, t,S=1,  fixed =False):
        # we do local centering and update of the mu's
        pred = np.zeros(data.N)
        for k in range(data.K):
            csi_k = self.a[k] + self.mu
           
            # update mu
            self.mu = np.random.normal((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]))
            sum_a_l = sum_on_all_new(self.a, data,k)
            csi_k = self.a[k] + self.mu          
            for s in range(S):
                x = np.random.multivariate_normal(csi_k, np.identity(len(csi_k))*np.sqrt(2)) 
                for i in range(len(csi_k)):
                    acc_1 = - self.tau_e/2*(csi_k[i]-x[i])*(2*(data.N_s[k][i]*data.mean_y_s[k][i] - sum_a_l[i])-(x[i]+csi_k[i])*data.N_s[k][i] )
                    acc_2 = - self.tau[k]/2*(x[i]-csi_k[i])*(x[i]+csi_k[i]-2*self.mu)
                    log_acc = acc_1+acc_2 
                    if np.log(np.random.rand())< log_acc: 
                        csi_k[i] = x[i]
            self.a[k] = csi_k - self.mu
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            if fixed == False:
                self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
            self.mu_chain[1+t*data.K+k]=self.mu
            pred += self.a[k][data.ii[:,k]]
        pred += self.mu

        if fixed == False:
            SS0 = np.sum(np.power((data.y-pred),2))
            self.tau_e =  np.random.gamma(size=1, shape= data.N/2-0.5, scale=2/SS0)
        self.tau_chain[:, t+1] = self.tau
        self.tau_e_chain[t+1] = self.tau_e
        self.a_means_chain[:, t+1] = self.a_means


# for the moment there's no randomness on mu, just take it equal to what i give
# for the moment there's no randomness on mu, just take it equal to what i give
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
        
        # supposing uniform assignment of individuals to factor/levels   
        self.ii = np.random.randint(0,self.I, size=(self.N,self.K)) # this should go from 0 to I-1
        self.iss = np.max(self.ii, axis = 0)+1 # note that for iss the categories go from 1 to I
        
        # self.mu = np.random.normal(mu_0, 1/tau_0)
        self.mu = mu
        #random effect, tau_k = 1
        self.a = np.random.normal(0,1/np.sqrt(tau_k), size = (self.I,self.K))
        aux = np.sum(self.a[self.ii,np.arange(self.K)], axis = 1) 
        # let's suppose that the y's are bernoulli's with probability 
        # of success given by probit transformations 
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
        

    
#initializing the values for the chain, and update them
class iter_value_laplace:  
    l = 2.38 #value for sampling
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
        
    
  
    def update(self, data, t, mask,S, fixed =False, save_aux =False, proposal_std= -1):
        # we do local centering and update of the mu's    
        for k in range(data.K):
            csi_k = self.a[k] + self.mu
            self.mu = np.random.normal((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]))
            
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
                  
                #acc_prob[k,:] = (t*acc_prob[k,:] + aux/S)/(t+1)
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

        
        # remember that proposal_var_cov should be a list of array! since all components are conditionally iid
    def coupled_update(self, val_2, data,t,l,mask,b,S, fixed = False, close = False, factorized_proposals = False, factorized_acceptance = True, a_pvb= True, a_pvf = True):        
        for k in range(data.K):
            # do centering
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
                # before modification 22/05
                #proposal_var_b = 1/(data.N_s[k]*(1/(2*b**2))+self.tau[k])*2.38**2/len(self.a[0])
                proposal_var_b = 1/(data.N_s[k]*(1/(2*b**2))+self.tau[k])*2 
            else:
                #general rescaling with dimension 
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
            
            # update precisions
            if fixed==False:
                self.tau[k], val_2.tau[k] = funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5,  2/np.sum(np.power(self.a[k],2)),  2/np.sum(np.power(val_2.a[k],2)))
     
            self.mu_chain[t*data.K+1+k]=self.mu
            val_2.mu_chain[t*data.K+1+k]=val_2.mu
            
        self.tau_chain[:, t+1] = self.tau
        self.a_means_chain[:, t+1] = self.a_means
        
        val_2.tau_chain[:, t+1] = val_2.tau
        val_2.a_means_chain[:, t+1] = val_2.a_means
        #if save_aux == True:
        #    for en,(k,i) in enumerate(self.list_saving):
        #        self.a_aux[en,t+1] = self.a[k][i]
        #        val_2.a_aux[en,t+1] = val_2.a[k][i]

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
    




#initializing the values for the chain, and update them
class iter_value_lapl:  
    l = 2.38 #value for sampling
    def __init__(self, data,T,a, mu_pr = 0, tau_pr = 0, rand = True):
        self.mu_pr = mu_pr
        self.tau_pr = tau_pr
        if rand == False:
            self.tau: numpy.ndarray = tau_k+0
            #self.mu =  mu+0
            #self.a = [data.a[:,k]+0 for k in range(data.K)]
            self.mu: float  =  np.random.normal(mu_pr,0.3)
            self.a: list = [np.random.normal(0,2, size = (data.iss[k])) for k in range(data.K)]
            
        else: 
            self.tau: numpy.ndarray = np.random.gamma(1,2,size = data.K)
            self.mu: float =  np.random.normal(mu_pr,0.3)
            self.a: list = [np.random.normal(0,2, size = (data.iss[k])) for k in range(data.K)] #otherwise variances explodes...
        #self.a = [np.random.normal(0,2, size = (data.iss[k])) for k in range(data.K)] #otherwise variances explodes...
        self.a_means: numpy.ndarray = np.array([self.a[k]@data.N_s[k]/data.N for k in range(data.K)])
        
        self.mu_chain = np.zeros(1+T*data.K)
        self.a_means_chain = np.zeros((data.K,T+1 ))
        self.tau_chain = np.zeros((data.K,T+1 ))
        self.mu_chain[0]= self.mu
        self.tau_chain[:,0] = self.tau
        self.a_means_chain[:,0]= self.a_means
        self.a_aux =  np.zeros(T+1)
        self.a_aux[0] = a[0][0]
        
    
    def MALA_update(self, data, t, mask, S, fixed = False):
        acc = 0  
        for k in range(data.K):
            # acc rate should be around 0.574 
            step = 0.01
            
            # do centering
            csi_k = self.a[k] + self.mu
            self.mu = np.random.normal((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]))
            old_all = [np.array(data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(K)], axis = 0))+  self.a[k][i] for i in range(len(csi_k))]
            for s in range(S):

                aux_m = [old_all[i]-  csi_k[i] for i in range(len(csi_k))]
                grad_k = np.array([np.sum(np.sign(aux_m[i]))/b for i in range(len(csi_k))]) -self.tau[k]*(csi_k-self.mu)
                z = np.random.normal(0, 1, size = len(csi_k))
                
                mean = csi_k + step* grad_k 
                x = mean + z*np.sqrt(2*step)
                
                aux_mt = [old_all[i] -x[i] for i in range(len(csi_k))]
                grad_kt = np.array([np.sum(np.abs(aux_mt[i]))/b for i in range(len(csi_k))]) -self.tau[k]/2*(x-self.mu)**2
                mean_t = x + step* grad_kt 
                #print([len(aux_m[i]) for i in range(len(csi_k))], [len(aux_m2[i]) for i in range(len(csi_k))],[len(aux_mt[i]) for i in range(len(csi_k))], [len(aux_mt2[i]) for i in range(len(csi_k))])
                
                for i in range(len(csi_k)):
                    acc_1 = - (np.sum(np.abs(aux_mt[i]))-np.sum(np.abs(aux_m[i])))/b 
                    acc_2 = - self.tau[k]/2* ((x[i]-self.mu)**2-(csi_k[i]-self.mu)**2)
                    #acc_3 = - step**2*m_kt[i]**2
                    acc_3 = - ((csi_k[i]-mean_t[i])**2 - (x[i]-mean[i])**2 )/(4*step)
                    
                    #log_acc = acc_1+acc_2 + acc_3
                    log_acc = acc_1 + acc_3 + acc_2
                    
                    if np.log(np.random.rand())< log_acc: 
                        csi_k[i] = x[i]
                        acc +=1

            
            self.a[k] = csi_k - self.mu
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            
            if fixed == False:
                self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
            self.mu_chain[1+t*data.K+k]=self.mu
        
        acc_prob[t+1] = acc/(I*S)
        self.a_aux[t+1] = self.a[0][0]
        self.tau_chain[:, t+1] = self.tau
        self.a_means_chain[:, t+1] = self.a_means

    def MALA_coupled_update(self, val_2, data,t,l,mask,S, fixed = False, close = False):
        for k in range(data.K):
            # acc rate should be around 0.574 
            step = 0.5
            # do centering
            csi_k = self.a[k] + self.mu
            csi_k2 = val_2.a[k] + val_2.mu
            
            mu_mean =   (self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k])
            mu_mean2 =  (self.tau_pr*self.mu_pr + val_2.tau[k]*np.sum(csi_k_2))/(self.tau_pr+val_2.tau[k]*data.iss[k])
            
            std_1 = 1/np.sqrt(tau_pr+self.tau[k]*data.iss[k])
            std_2 = 1/np.sqrt(tau_pr+val_2.tau[k]*data.iss[k])
            
            if close:
                self.mu, val_2.mu = funct_simulation.maximal_coupling_normal_1d(mu_mean, mu_mean2, std_1, std_2)
            else:
                z = np.random.normal(0,1)
                self.mu = mu_mean + z*std_1
                val_2.mu = mu_mean2 + z*std_2
            
            old_all = [np.array(data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(K)], axis = 0))+  self.a[k][i] for i in range(len(csi_k))]
            old_all2 = [np.array(data.y[mask[k][i]] - np.sum([val_2.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(K)], axis = 0))+  val_2.a[k][i] for i in range(len(csi_k))]
            
            for s in range(S):
                

                
                aux_m = [old_all[i]-  csi_k[i] for i in range(len(csi_k))]
                aux_m = [old_all2[i]-  csi_k2[i] for i in range(len(csi_k))]
                m_k = np.array([-np.sum(aux_m[i])/b for i in range(len(csi_k))]) -self.tau[k]/2*(csi_k-self.mu)**2
                m_k2 = np.array([-np.sum(aux_m2[i])/b for i in range(len(csi_k))]) -val_2.tau[k]/2*(csi_k2-val_2.mu)**2
                
                
                if close:
                    x,x2= funct_simulation.reflection_coupling(csi_k + step* m_k, csi_k2 + step* m_k2, np.identity(len(csi_k))*2*step, diag = True)
                    
                else:
                    z = np.random.normal(0, 1, size = len(csi_k))
                    x = csi_k + step* m_k + z*np.sqrt(2*step)
                    x2 = csi_k2 + step* m_k2 + z*np.sqrt(2*step)
                
                aux_mt = [old_all[i] -x[i] for i in range(len(csi_k))]
                aux_mt2 = [old_all2[i] -x2[i] for i in range(len(csi_k))]
                m_kt = np.array([-np.sum(aux_mt[i])/b for i in range(len(csi_k))]) -self.tau[k]/2*(x-self.mu)**2
                m_kt2 = np.array([-np.sum(aux_mt2[i])/b for i in range(len(csi_k))]) -val_2.tau[k]/2*(x2-val_2.mu)**2
                
                #print([len(aux_m[i]) for i in range(len(csi_k))], [len(aux_m2[i]) for i in range(len(csi_k))],[len(aux_mt[i]) for i in range(len(csi_k))], [len(aux_mt2[i]) for i in range(len(csi_k))])
                
                for i in range(len(csi_k)):
                    acc_1 = - (np.sum(np.abs(aux_mt[i]))-np.sum(np.abs(aux_m[i])))/b
                   # acc_2 = - self.tau[k]/2*((x-csi_k[i])*(x+csi_k[i]-2*self.mu))
                    acc_3 = - step**2*m_kt[i]**2
                    
                    acc_12 = - (np.sum(np.abs(aux_mt2[i]))-np.sum(np.abs(aux_m2[i])))/b
                   # acc_2 = - self.tau[k]/2*((x-csi_k[i])*(x+csi_k[i]-2*self.mu))
                    acc_32 = - step**2*m_k2t[i]**2
                    
                    
                    #log_acc = acc_1+acc_2 + acc_3
                    log_acc = acc_1 + acc_3
                    log_acc2 = acc_12 + acc_32
                    
                    u = np.log(np.random.rand())
                    if u < log_acc: 
                        csi_k[i] = x[i]
                    
                    if u < log_acc: 
                        csi_k2[i] = x2[i]

            
            self.a[k] = csi_k - self.mu
            val_2.a[k] = csi_k2 - val_2.mu
            
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            val_2.a_means[k] =  val_2.a[k]@data.N_s[k]/data.N

            # update precisions
            if fixed==False:
                self.tau[k], val_2.tau[k] = funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5,  2/np.sum(np.power(self.a[k],2)),  2/np.sum(np.power(val_2.a[k],2)))
                
            self.mu_chain[t*data.K+1+k]=self.mu
            val_2.mu_chain[t*data.K+1+k]=val_2.mu
            
        self.tau_chain[:, t+1] = self.tau
        self.a_means_chain[:, t+1] = self.a_means
        
        val_2.tau_chain[:, t+1] = val_2.tau
        val_2.a_means_chain[:, t+1] = val_2.a_means
        self.a_aux[t+1] = self.a[0][0]
        val_2.a_aux[t+1] = val_2.a[0][0]

        
        if (self.mu == val_2.mu) and np.array([(self.a[k] == val_2.a[k]).all() for k in range(data.K)]).all() and (self.tau == val_2.tau).all():
            print("yeeeee")           
            return -1

        
        
        
    def update(self, data, t, mask,S, fixed =False):
        # we do local centering and update of the mu's    
        for k in range(data.K):
            
            # do centering
            csi_k = self.a[k] + self.mu
            prop_dvst = 1
            # update mu
            self.mu = np.random.normal((self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k]),1/np.sqrt(self.tau_pr+self.tau[k]*data.iss[k]))
            #update of csi_k, component by component
            num = 500
            b = data.b
            
            old_all = [np.array(data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(data.K)], axis = 0))+  self.a[k][i] for i in range(len(csi_k))]
            for s in range(S):
                aux_m = [old_all[i]-  csi_k[i] for i in range(len(csi_k))]
                c_ik = np.array([1/(self.tau[k] + num*data.N_s[k][i]/b -num*np.sum((np.tanh(num*aux_m[i]))**2)/b)   for i in range(len(csi_k))])
                
                z = np.random.normal(0, 1, size = len(csi_k))
                x = csi_k + z*c_ik
                aux_mt = [old_all[i] -x[i] for i in range(len(csi_k))]
                c_ikt = np.array([1/(self.tau[k] + num*data.N_s[k][i]/b -num*np.sum((np.tanh(num*aux_mt[i]))**2)/b)   for i in range(len(csi_k))])
                
                #print([len(aux_m[i]) for i in range(len(csi_k))], [len(aux_m2[i]) for i in range(len(csi_k))],[len(aux_mt[i]) for i in range(len(csi_k))], [len(aux_mt2[i]) for i in range(len(csi_k))])
                
                for i in range(len(csi_k)):
                    acc_1 = - (np.sum(np.abs(aux_mt[i]))-np.sum(np.abs(aux_m[i])))/b
                   # acc_2 = - self.tau[k]/2*((x-csi_k[i])*(x+csi_k[i]-2*self.mu))
                    acc_3 = - (c_ikt[i]/2 -c_ik[i]/2) *(csi_k[i]- x[i])**2
                    
                    #log_acc = acc_1+acc_2 + acc_3
                    log_acc = acc_1 + acc_3
                    
                    if np.log(np.random.rand())< log_acc: 
                        csi_k[i] = x[i]
            """
                        
            
            old_all = [ np.array(data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(K)], axis = 0))+  self.a[k][i] for i in range(len(csi_k))]
                        
            for i in range(len(csi_k)):
                auxi = 0
                # do some steps of metropolis
                for s in range(S):
                    aux_m = old_all[i] -  csi_k[i] # questo cambia ad ogni update
                    c_ik = 1/(self.tau[k] + num*data.N_s[k][i]/b -num*np.sum((np.tanh(num*aux_m))**2)/b) 
                    # m_ik = ((c_ik**-1-self.tau[k])*csi_k[i] +np.sum(np.sign(aux_m))/b + self.mu*self.tau[k])*c_ik
                    x = np.random.normal(csi_k[i], np.sqrt(c_ik))
                    aux_mt = old_all[i] -x
                    # c_ikt = 1/(self.tau[k] +num*data.N_s[k][i]/b -num*np.sum((np.tanh(num*aux_mt))**2)/b) 
                    # m_ikt = ((c_ikt**-1-self.tau[k])*x+np.sum(np.sign(aux_mt))/b + self.mu*self.tau[k])*c_ikt
                    acc_1 = - (np.sum(np.abs(aux_mt))-np.sum(np.abs(aux_m)))/b
                   # acc_2 = - self.tau[k]/2*((x-csi_k[i])*(x+csi_k[i]-2*self.mu))
                   # acc_3 = - c_ikt/2*(csi_k[i]-m_ikt)**2 + c_ik/2*(x-m_ik)**2 
                    
                    #log_acc = acc_1+acc_2 + acc_3
                    log_acc = acc_1
                    
                    if np.log(np.random.rand())< log_acc: 
                        csi_k[i] = x
                        auxi += 1
                        """
                #acc_prob[k,i] = ((t) * acc_prob[k,i] + auxi /S)/(t+1)
            
            self.a[k] = csi_k - self.mu

            
            #acc_prob[k,t] = auxi/(len(csi_k)*S)
            
            self.a_means[k] = self.a[k]@data.N_s[k]/data.N
            if fixed == False:
                self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/np.sum(np.power(self.a[k],2)))
            self.mu_chain[1+t*data.K+k]=self.mu
        
        self.a_aux[t+1] = self.a[0][0]
        self.tau_chain[:, t+1] = self.tau
        self.a_means_chain[:, t+1] = self.a_means


        
    # always try coupling, no distinction near/close
    def coupled_update(self, val_2, data,t,l,mask,S, fixed = False, close = False):
        num = 500
        l = 2.38
        
        
        for k in range(data.K):
            # do centering
            csi_k = self.a[k] + self.mu
            csi_k_2 = val_2.a[k] + val_2.mu
            
            mu_mean =   (self.tau_pr*self.mu_pr + self.tau[k]*np.sum(csi_k))/(self.tau_pr+self.tau[k]*data.iss[k])
            mu_mean_2 = (self.tau_pr*self.mu_pr + val_2.tau[k]*np.sum(csi_k_2))/(self.tau_pr+val_2.tau[k]*data.iss[k])
            
            std_1 = 1/np.sqrt(tau_pr+self.tau[k]*data.iss[k])
            std_2 = 1/np.sqrt(tau_pr+val_2.tau[k]*data.iss[k])
            
  
            self.mu, val_2.mu = funct_simulation.maximal_coupling_normal_1d(mu_mean, mu_mean_2, std_1, std_2)
            
            #else:
            #    aux = np.random.normal(0,1)
            #    self.mu = mu_mean + aux*std_1
            #    val_2.mu = mu_mean_2 + aux*std_2
                
             
            old_all =  ([np.array(data.y[mask[k][i]] - np.sum([self.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(K)], axis = 0))+  self.a[k][i] for i in range(len(csi_k))])
            old_all_2= ([np.array(data.y[mask[k][i]] - np.sum([val_2.a[cappa][data.ii[mask[k][i],cappa]] for cappa in range(K)], axis = 0))+  val_2.a[k][i] for i in range(len(csi_k))])# dimensione corretta

            
            
            for s in range(S):
                aux_m = [old_all[i]-  csi_k[i] for i in range(len(csi_k))]
                c_ik = np.array([1/(self.tau[k] + num*data.N_s[k][i]/b -num*np.sum((np.tanh(num*aux_m[i]))**2)/b)   for i in range(len(csi_k))])
                
                aux_m2 = [old_all_2[i] -  csi_k_2[i] for i in range(len(csi_k))]  # questo cambia ad ogni update
                c_ik2 = np.array([1/(val_2.tau[k] + num*data.N_s[k][i]/b -num*np.sum((np.tanh(num*aux_m2[i]))**2)/b)  for i in range(len(csi_k))])
                
                prop_dvst = l/np.sqrt(len(csi_k)) 
                
         
                if close:
                    x,y= funct_simulation.reflection_coupling(csi_k,csi_k_2, np.identity(len(csi_k))*prop_dvst**2, diag = True)
                else:
                    z = np.random.normal(0,1, size = len(csi_k))
                    x = csi_k + z*c_ik
                    y = csi_k_2 +z*c_ik2
                
                aux_mt = [old_all[i] -x[i] for i in range(len(csi_k))]
                aux_mt2 =[old_all_2[i] -y[i] for i in range(len(csi_k))]
                
                
                #print([len(aux_m[i]) for i in range(len(csi_k))], [len(aux_m2[i]) for i in range(len(csi_k))],[len(aux_mt[i]) for i in range(len(csi_k))], [len(aux_mt2[i]) for i in range(len(csi_k))])
                
                for i in range(len(csi_k)):
            
                   # m_ik = ((c_ik**-1-self.tau[k])*csi_k[i] +np.sum(np.sign(aux_m))/b + self.mu*self.tau[k])*c_ik

                    #m_ik2 = ((c_ik2**-1-val_2.tau[k])*csi_k_2[i] +np.sum(np.sign(aux_m2))/b + val_2.mu*val_2.tau[k])*c_ik2

                    
                    
                    #c_ikt = 1/(self.tau[k] +num*data.N_s[k][i]/b - np.sum((np.tanh(aux_mt))**2)/b) 
                    #m_ikt = ((c_ikt**-1-self.tau[k])*x+np.sum(np.sign(aux_mt))/b + self.mu*self.tau[k])*c_ikt
                    
                    
                    #c_ikt2 = 1/(val_2.tau[k] +num*data.N_s[k][i]/b - np.sum((np.tanh(aux_mt2))**2)/b) 
                    #m_ikt2 = ((c_ikt2**-1-val_2.tau[k])*y+np.sum(np.sign(aux_mt2))/b + val_2.mu*val_2.tau[k])*c_ikt2
                    
                    acc_1 = - (np.sum(np.abs(aux_mt[i]))-np.sum(np.abs(aux_m[i])))/b
                    #acc_2 = - self.tau[k]/2*((x-csi_k[i])*(x+csi_k[i]-2*self.mu))
                    #acc_3 = - c_ikt/2*(csi_k[i]-m_ikt)**2 + c_ik/2*(x-m_ik)**2 
                    
                    acc_12 = - (np.sum(np.abs(aux_mt2[i]))-np.sum(np.abs(aux_m2[i])))/b
                    #acc_22 = - val_2.tau[k]/2*((y-csi_k_2[i])*(y+csi_k_2[i]-2*val_2.mu))
                    #acc_32 = - c_ikt2/2*(csi_k_2[i]-m_ikt2)**2 + c_ik2/2*(y-m_ik2)**2 
                    
                    
                    #log_acc = acc_1+acc_2 + acc_3
                    log_acc = acc_1
                    #log_acc2 = acc_12+acc_22 + acc_32
                    log_acc2 = acc_12
                    u = np.log(np.random.rand())
                    if u < log_acc: 
                        csi_k[i] = x[i]
                        
                    if u < log_acc2: 
                        csi_k_2[i] = y[i]

            
                """            for s in range(S):
                
                   
                
                aux_m = [old_all[i] - csi_k[i] for i in range(len(csi_k))]
                c_k = [ 1/(self.tau[k] + num/b*np.sum(1-(np.tanh(aux_m[i]))**2) ) for i in range(len(csi_k))]
                m_k = csi_k+ [(1/b* np.sum(np.sign(aux_m[i])) - self.tau[k]*(csi_k[i]-self.mu))*(c_k[i]) for i in range(len(csi_k))]

                aux_m_2 = [old_all_2[i] - csi_k_2[i] for i in range(len(csi_k))]
                c_k_2 = [ 1/(val_2.tau[k] + num/b*np.sum(1-(np.tanh(aux_m_2[i]))**2) ) for i in range(len(csi_k))]
                m_k_2 = csi_k_2+ [(1/b* np.sum(np.sign(aux_m_2[i])) - val_2.tau[k]*(csi_k_2[i]-val_2.mu))*(c_k_2[i]) for i in range(len(csi_k))]

                
                aux_mt = [old_all[i] - x[i] for i in range(len(csi_k))]
                aux_mt_2 = [old_all_2[i] - y[i] for i in range(len(csi_k))]
                    
                c_kt = [ 1/(self.tau[k] + num/b*np.sum(1-(np.tanh(aux_mt[i]))**2) ) for i in range(len(csi_k))]
                c_kt_2 = [ 1/(val_2.tau[k] + num/b*np.sum(1-(np.tanh(aux_mt_2[i]))**2) ) for i in range(len(csi_k))]
                    
                m_kt = csi_k+ [(1/b* np.sum(np.sign(aux_mt[i])) - self.tau[k]*(x[i]-self.mu))*(c_kt[i]) for i in range(len(csi_k))]
                m_kt_2 = csi_k_2+ [(1/b* np.sum(np.sign(aux_mt_2[i])) - val_2.tau[k]*(y[i]-val_2.mu))*(c_kt_2[i]) for i in range(len(csi_k))]

                log_acc = - 1/b*np.array([np.sum(np.abs(old_all[i]-x[i]))-np.sum(np.abs(old_all[i]-csi_k[i])) for i in range(data.iss[k])]) - self.tau[k]/2*((x-csi_k)*(x+csi_k-2*self.mu)) - self.tau[k]/2*(csi_k-m_kt)**2 + self.tau[k]/2*(x-m_k)**2 
                log_acc_2 = - 1/b*np.array([np.sum(np.abs(old_all_2[i]-y[i]))-np.sum(np.abs(old_all_2[i]-csi_k_2[i])) for i in range(data.iss[k])]) - val_2.tau[k]/2*((y-csi_k_2)*(y+csi_k_2-2*val_2.mu)) - val_2.tau[k]*(csi_k_2-m_kt_2)**2 + val_2.tau[k]/2*(y-m_k_2)**2 
  
                rand_u = np.log(np.random.rand())
    
                for i in range(len(csi_k)):
                        #log_acc = - 1/b*(np.sum(np.abs(old_all[i]-x[i]))-np.sum(np.abs(old_all[i]-csi_k[i]))) - self.tau[k]/2*((x[i]-csi_k[i])*(x[i]+csi_k[i]-2*self.mu)) - self.tau[k]/2*(csi_k[i]-m_kt[i])**2 + self.tau[k]/2*(x[i]-m_k[i])**2 
                        #log_acc_2 = - 1/b*(np.sum(np.abs(old_all_2[i]-y[i]))-np.sum(np.abs(old_all_2[i]-csi_k_2[i]))) - val_2.tau[k]/2*((y[i]-csi_k_2[i])*(y[i]+csi_k_2[i]-2*val_2.mu)) - val_2.tau[k]*(csi_k_2[i]-m_kt_2[i])**2 + val_2.tau[k]/2*(y[i]-m_k_2[i])**2 

                    if rand_u< log_acc[i]: 
                        csi_k[i] = x[i]
    #                        auxi+=1
                    if rand_u< log_acc_2[i]: 
                        csi_k_2[i] = y[i]
                """

            self.a[k] = csi_k - self.mu
            val_2.a[k] = csi_k_2 - val_2.mu
            
            self.a_means[k] =  self.a[k]@data.N_s[k]/data.N
            val_2.a_means[k] =  val_2.a[k]@data.N_s[k]/data.N
            
            # update precisions
            if fixed==False:
                self.tau[k], val_2.tau[k] = funct_simulation.maximal_coupling_gamma(data.iss[k]/2-0.5,  2/np.sum(np.power(self.a[k],2)),  2/np.sum(np.power(val_2.a[k],2)))
                
                #else:
                #    self.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/sum(np.power(self.a[k],2)))               
#
                #    val_2.tau[k] = np.random.gamma(size=1, shape= data.iss[k]/2-0.5, scale= 2/sum(np.power(val_2.a[k],2)))               

     
            self.mu_chain[t*data.K+1+k]=self.mu
            val_2.mu_chain[t*data.K+1+k]=val_2.mu
            
        self.tau_chain[:, t+1] = self.tau
        self.a_means_chain[:, t+1] = self.a_means
        
        val_2.tau_chain[:, t+1] = val_2.tau
        val_2.a_means_chain[:, t+1] = val_2.a_means
        self.a_aux[t+1] = self.a[0][0]
        val_2.a_aux[t+1] = val_2.a[0][0]

        
        if (self.mu == val_2.mu) and np.array([(self.a[k] == val_2.a[k]).all() for k in range(data.K)]).all() and (self.tau == val_2.tau).all():
            print("yeeeee")           
            return -1
        
       
    
