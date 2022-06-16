#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[13]:


#nos of inputs, nos of steps is Δt, T=time, r= Int. rate, sigma= volatility,S_0=initital value of   

def GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):
    
    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps]) # standard rand no. [matrix with nos of paths & nos of steps]
    X = np.zeros([NoOfPaths,NoOfSteps+1]) # process of ND
    S = np.zeros([NoOfPaths,NoOfSteps+1]) # exponent of X
    time = np.zeros([NoOfSteps+1])
    
    X[:,0] = np.log(S_0) # first coloum of matrix  X0= log return of initial stock
    
    dt = T / float(NoOfSteps) #dt=	Δt/steps to simulate our paths
    for i in range(0,NoOfSteps):
        #making sure the samples is in N.D it helps us to achieve better convergence
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
     #for next coloum of matrix we will do
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma **2 ) * dt + sigma * np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
   
    #Compute exponent of AM
    S = np.exp(X)
    paths = {"time":time,"X":X,"S":S}
    return paths


def mainCalculation():
    NoOfPaths = 5000
    NoOfSteps = 500
    T = 1
    r = 0.05
    sigma = 0.4
    S_0 = 100
    
    #Money saving account
    M = lambda r,t: np.exp(r*t)

    Paths = GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    timeGrid = Paths["time"]
    X = Paths["X"]
    S = Paths["S"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(S))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    
    # checking martingale property by expectation of S
    ES = np.mean(S[:,-1]) #-1 means take mean of last coloum
    print(ES)
    
    #we will check discounted stock value 
    #We take FV of stock divided by money saving acc at the maturity
    ESM = np.mean(S[:,1]/M(r,T))
    print(ESM)
    
mainCalculation()


# In[31]:





def GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    
    # Fixing random seed
    #np.random.seed(1)
        
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = np.log(S_0)
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma **2 ) * dt + sigma * np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
        
    #Compute exponent of ABM
    S = np.exp(X)
    paths = {"time":time,"X":X,"S":S}
    return paths

def mainCalculation():
    NoOfPaths = 500
    NoOfSteps = 500

    T = 1
    r = 0.05
    sigma = 0.4
    S_0 = 100
    
    M = lambda r,t: np.exp(r*t)
    
    Paths = GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    timeGrid = Paths["time"]
    X = Paths["X"]
    S = Paths["S"]
    
    #plt.figure(1)
    #plt.plot(timeGrid, np.transpose(X))   
    #plt.grid()
    #plt.xlabel("time")
    #plt.ylabel("X(t)")
    
    #plt.figure(2)
    #plt.plot(timeGrid, np.transpose(S))   
    #plt.grid()
    #plt.xlabel("time")
    #plt.ylabel("S(t)")
    
    # checking martingale property
    ES = np.mean(S[:,-1])
    print(ES)
    
    ESM = np.mean(S[:,-1]/M(r,T))
    print(ESM)
    
mainCalculation()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




