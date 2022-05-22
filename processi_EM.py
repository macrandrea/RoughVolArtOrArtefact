import numpy as np
import scipy.stats as ss
from scipy.optimize import fsolve
from math import log1p as log


#23400 secondi nella giornata
#alcune vecchie simulazioni hanno np.random.standard_normal(), ma np.random.randn() funzion allo stesso modo

def RV(o,step):
    r=np.zeros(step)
    s=np.zeros(step)
    for i in range(1,len(o)):
        r[i]=np.sqrt(log(o[i]/o[i-1])**2)
    for i in range(1,len(o)):
        s[i]=np.std(r[i:i+2])*np.sqrt(step)
    return s

def RV_5min(o,step,N):
    r=np.zeros(step)
    s=np.zeros(step)
    for i in range(0,len(o)):
        r[i]=np.sqrt(np.log(o[i]/o[i-1])**2)
    for i in range(0,len(r)):
        s[i]=np.std(r[i:i+2])*np.sqrt(N/step)
    return s

def gbm_mod(s0,T,N,seed=457778):#N=23400
    S=np.zeros(N)
    sigma=np.ones(N)
    sigma[0]=0.2
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    for i in range(1,len(S)):
        sigma[i] = np.abs(sigma[i-1]+(np.random.randn()*np.sqrt(dt)))
        S[i]=S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*np.random.randn()
    return [S,sigma]

def gbm_expOU(s0,T,N,seed=457778,gamma=1,alpha=1,theta=1,rho=-0.7,r=0):
    ''' corretta''' #giusto
    S=np.zeros(N)
    sigma=np.ones(N)
    y=np.ones(N)
    y[0]=0
    sigma[0]=1
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    cov=np.matrix([[1, rho],[rho, 1]])
    a  =np.linalg.cholesky(cov)
    for i in range(1,len(S)):
        epsilon1=np.random.randn()*a[0,0]
        epsilon2=np.random.randn()*a[1, 0]+ np.random.randn() * a[1, 1 ]       
        S[i] = S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1
        y[i] = y[i-1] -gamma*y[i-1]*dt+ theta*np.sqrt(dt)*epsilon2
        sigma[i] = np.exp(y[i])
    return [S,sigma]

def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, Npaths, return_vol=False):
    dt = T/steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0,0]), 
                                           cov = np.array([[1,rho],
                                                          [rho,1]]), 
                                           size=Npaths) * np.sqrt(dt) 
        
        S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t) *WT[:,0] ) ) 
        v_t = np.abs(v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t)*WT[:,1])
        prices[:, t] = S_t
        sigs[:, t] = v_t
    
    if return_vol:
        return prices, sigs
    
    return prices


def gbm_expOUnoCor(s0,T,N,seed=457778,gamma=1,alpha=1,theta=1,rho=-0.7,r=0):
    ''' corretta''' #giusto
    S=np.zeros(N)
    sigma=np.ones(N)
    y=np.ones(N)
    y[0]=0
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    cov=np.matrix([[1, rho],[rho, 1]])
    a  =np.linalg.cholesky(cov)
    for i in range(1,len(S)):
        epsilon1=np.random.randn()#*a[0,0]
        epsilon2=np.random.randn()#*a[1, 0]+ np.random.randn() * a[1, 1 ]       
        S[i] = S[i-1] +S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1
        y[i] = y[i-1] +alpha*(gamma-y[i-1])*dt+ theta*np.sqrt(dt)*epsilon2
        sigma[i] = np.exp(y[i-1])
    return [S,sigma]


def calc_delta(sigma,rv,step,N):
    D=np.zeros(N)
    for (i,j) in zip(range(1,N,step),range(1,np.int8(N/step))):
        D[i]=sigma[i]-rv[j]
        d=list(filter(lambda a: a != 0, D))
    return d

def slice_stock(x):#s[0],23400,300):
    v=[]
    for i,ii in zip(x[:-1],x[1:]):
        v.append([i,ii])
    return v  

def campionamento(x,N,step):
    s=np.zeros(np.int8(N/step))
    for j,i in zip(range(0,N,step),range(0,np.int8(N/step))):
        s[i]=x[j]
    return s

def calcolaRendimenti(s,N):
    l=np.zeros(N)
    for i in range(1,N):
        l[i]=np.abs(log(s[i])-log(s[i-1]))
    return l

def realVol(x,step,N):
    y=np.zeros(np.int8(N/step))
    iter=step
    iter2=0
    for i in range(1,len(y)):
        y[i]=sum(x[iter2:iter])#*np.sqrt(78)
        iter2=iter-step
        iter =iter+step
    return y

#def calc_W():
#    def ls():
#        return np.norm(W-T)^2 #griglia di p e prendi il minimo per norma quadrata
#    w=fsolve(ls)