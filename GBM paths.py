#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[61]:


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
    NoOfPaths = 25
    NoOfSteps = 500
    T = 1
    r = 0.05
    sigma = 0.4
    S_0 = 100

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
    
mainCalculation()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




