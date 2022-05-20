import numpy as np
import scipy.stats as ss
from scipy.optimize import fsolve

def slice_stock(X,N,step):
    #logret= []
    #RV= np.zeros(np.int8(step/N))
    o=[]#np.zeros(np.int8(step/N))
    #for i in range(1,len(logret),step):
    for i in range(0,N,step):
        o.append(X[i])
        #logret.append(np.sqrt(np.log(o[i]/o[i-1])**2))
    #for j in range(78):
    #    RV[j]= np.linalg.norm(np.log(o[j])-np.log(o[j-1]))*np.sqrt(78)
        #o[i]=np.linalg.norm(np.log(X[i])-np.log(X[i-1]))*np.sqrt(N)
        #logret[i]=np.sqrt(np.log(X[i]/X[i-1])**2)
        #RV[i]= np.sqrt(logret[i])*np.sqrt(N) #-logret[i-1] #è gia una realized volatility
        #rv=list(filter(lambda a: a != 0, RV))
        #for j,ii in zip(logret[:-1],logret[1:]):
        #    o.append(np.std([j,ii])*np.sqrt(step))        
    return o#RV#rv

#for i,ii in zip(p[:-1],p[1:]):
#    o.append(np.std([i,ii])*np.sqrt(1/15))
#o
def RV(o):
    RV=np.zeros(78)
    for j in range(78):
        RV[j]= np.linalg.norm(np.log(o[j])-np.log(o[j-1]))*np.sqrt(78)
    return RV

def gbm_mod(s0,T,N,seed=457778):#N=23400
    S=np.zeros(N)
    sigma=np.ones(N)
    sigma[0]=0.2
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    for i in range(1,len(S)):
        sigma[i] = np.abs(sigma[i-1]+(np.random.standard_normal()*np.sqrt(dt)))
        #sigma[i]=sigma[i-1]+np.sqrt(np.abs(np.random.standard_normal())*dt)
        S[i]=S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*np.random.standard_normal()
    return [S,sigma]


 
def gbm_OU(s0,T,N,seed=457778,gamma=0,alpha=1,theta=1,rho=0.7,r=0):
    S=np.zeros(N)
    sigma=np.ones(N)
    y=np.ones(N)
    y[0]=0.2
    sigma[0]=0.2
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    MU = np.array([0, 0])
    COV = np.matrix([[1, rho], [rho, 1]])
    for i in range(1,len(S)):
        rand = ss.multivariate_normal.rvs( mean=MU, cov=COV, size=N)
        W_y= rand[:,0]
        W_s= rand[:,1]
        sigma[i] = sigma[i-1] + gamma-(gamma-sigma[i-1])*dt+ theta*np.sqrt(dt)*W_y[i]
        y[i] = sigma[0]*np.exp(sigma[i])
        S[i] = S[i-1]+S[i-1]*y[i-1]*np.sqrt(dt)*W_s[i]
    return [S,y]

def calc_delta(sigma,rv,step,N):
    D=np.zeros(N)#np.int8(N/step))
    for (i,j) in zip(range(1,N,step),range(1,np.int8(N/step))):
        D[i]=sigma[i]-rv[j]
        d=list(filter(lambda a: a != 0, D))
    return d


#def calc_W():
#    def ls():
#        return np.norm(W-T)^2 #griglia di p e prendi il minimo per norma quadrata
#    w=fsolve(ls)