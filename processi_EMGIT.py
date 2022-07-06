import numpy as np
import scipy.stats as ss
from scipy.optimize import fsolve
from math import log1p as log
from fbm import FBM


#23400 secondi nella giornata
#alcune vecchie simulazioni hanno np.random.standard_normal(), ma np.random.randn() funzion allo stesso modo

def RV(o,step):
    r=np.zeros(len(o)+1)
    s=np.zeros(len(o)+1)
    #l=np.empty(step)
    #for i in range(len(o)):
    #    l[i]=np.log(o[i])
    for i in range(1,len(o)):
        r[i-1]=np.sqrt((log(o[i])-log(o[i-1]))**2)
    for i in range(0,len(r)):
        s[i]=np.sum(r[i:i+1])*np.sqrt(step)
    return s[:-2]

def RV_5min(o,step,N):
    r2=np.empty(N)
    s =np.empty(N)
    l=np.empty(N)
    for i in range(len(o)):
        l[i]=np.log(o[i])
    for i in range(1,len(o)):
        r2[i-1]=(l[i]-l[i-1])**2
    for i in range(0,len(r2)):
        s[i]=np.sqrt(np.sum(r2[i:i+2])*11700)
        #if s[i]<1e-4 or s[i]>0.6:
        #    s[i]=s[i-1]
    return s

def gbm_mod_adj(s0,T,N,seed=457778):#N=23400
    S=np.zeros(N)
    sigma=np.ones(N)
    sigma[0]=0.2
    S[0]=s0
    #omeghino=np.zeros(N)
    np.random.seed(seed)
    dt=T/N
    cov=np.matrix([[1, -0.79],[-0.79, 1]])
    a  =np.linalg.cholesky(cov)
    for i in range(1,len(S)):
        epsilon1=np.random.randn()
        epsilon2=np.random.randn()#*a[1, 0]+ np.random.randn() * a[1, 1 ] 
        sigma[i] = np.abs(sigma[i-1]+(epsilon2*np.sqrt(dt)))
        #omeghino[i]= (0.05**2)*np.sqrt(0.01*sigma[i])#0.01*(0.05**2*np.sqrt(np.exp(sigma[i]))*np.sqrt(dt)) #(0.05**2)*(0.01*np.sqrt(np.exp(sigma[i])))**4
        #1/N*np.sqrt(sum(0.01*np.exp(sigma))**4))
        eps=np.random.normal(scale=np.sqrt(sigma[i]))
        S[i]=(S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1)+eps
    #for i in range(1,len(S)):
        
        
        
    return [S,sigma]

def gbm_mod(s0,T,N,seed=457778):#N=23400
    S=np.zeros(N)
    sigma=np.ones(N)
    sigma[0]=0.2
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    cov=np.matrix([[1, -0.79],[-0.79, 1]])
    a  =np.linalg.cholesky(cov)
    for i in range(1,len(S)):
        epsilon1=np.random.randn()
        epsilon2=np.random.randn()#*a[1, 0]+ np.random.randn() * a[1, 1 ] 
        sigma[i] = np.abs(sigma[i-1]+(epsilon2*np.sqrt(dt)))
        S[i]=S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1
    return [S,sigma]

def gbm_expOU(s0,T,N,seed=457778,gamma=1,alpha=1,theta=1,rho=0,r=0):
    ''' corretta''' #giusto
    S=np.zeros(N)
    sigma=np.ones(N)
    y=np.ones(N)
    y[0]=0
    sigma[0]=1/100
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    cov=np.matrix([[1, rho],[rho, 1]])
    a  =np.linalg.cholesky(cov)
    for i in range(1,len(S)):
        epsilon1=np.random.randn()*a[0,0]
        epsilon2=np.random.randn()*a[1, 0]+ np.random.randn() * a[1, 1 ]       
        S[i] = S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1
        y[i] = (y[i-1] -gamma*y[i-1]*dt+ theta*np.sqrt(dt)*epsilon1)
        sigma[i] = np.exp(y[i])/100
    return [S,sigma]

def gbm_expOU_adj(s0,T,N,seed=457778,gamma=1,alpha=1,theta=1,rho=0,r=0):
    ''' corretta''' #giusto
    S=np.zeros(N)
    omeghino=np.zeros(N)
    sigma=np.ones(N)
    y=np.ones(N)
    y[0]=0
    sigma[0]=1/100
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    cov=np.matrix([[1, rho],[rho, 1]])
    a  =np.linalg.cholesky(cov)
    for i in range(1,len(S)):
        epsilon1=np.random.randn()*a[0,0]
        epsilon2=np.random.randn()*a[1, 0]+ np.random.randn() * a[1, 1 ]       
        y[i] = (y[i-1] -gamma*y[i-1]*dt+ theta*np.sqrt(dt)*epsilon1)
        sigma[i] = np.exp(y[i])/100
        eps=np.random.normal(scale=np.sqrt(sigma[i]))
        S[i] = (S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1)+eps
        
        #omeghino[i]= (0.05**2)*np.sqrt(0.01*np.sqrt(sigma[i])**4)#0.01*(0.05**2*np.sqrt(np.exp(sigma[i]))*np.sqrt(dt)) #(0.05**2)*(0.01*np.sqrt(np.exp(sigma[i])))**4
        #1/N*np.sqrt(sum(0.01*np.exp(sigma))**4))
    return [S,sigma]


def gbm_expOUsoloVol(s0,T,N,seed=457778,gamma=1,alpha=1,theta=1,rho=0,r=0):
    ''' corretta''' #giusto
    S=np.zeros(N)
    sigma=np.ones(N)
    y=np.ones(N)
    y[0]=0
    sigma[0]=1/100
    S[0]=s0
    np.random.seed(seed)
    dt=T/N
    cov=np.matrix([[1, rho],[rho, 1]])
    a  =np.linalg.cholesky(cov)
    for i in range(1,len(S)):
        epsilon1=np.random.randn()*a[0,0]
        epsilon2=np.random.randn()*a[1, 0]+ np.random.randn() * a[1, 1 ]       
        S[i] = S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1
        y[i] = (y[i-1] -gamma*y[i-1]*dt+ theta*np.sqrt(dt)*epsilon1)
        sigma[i] = np.exp(y[i])/100
    return sigma    


def rough(N,h,seed=457778):
    np.random.seed(seed)
    y=np.empty(N)
    s=np.empty(N)
    s[0]=100
    sigma=np.empty(N)
    sigma[0]=0.01
    dt=1/N
    cov=np.matrix([[1, -0.79],[-0.79, 1]])
    a  =np.linalg.cholesky(cov)
    h1=fbm(n=N, hurst=h, length=1, method='daviesharte')
    h2=fbm(n=N, hurst=0.5, length=1, method='daviesharte')
    for i in range(1,N):
        #y[i]    = y[i-1] -1*y[i-1]*dt+ 1*np.sqrt(dt)*h1[i-1]
        sigma[i]= np.exp(h1[i])/100#'''*a[1, 0]+h1[i] * a[1, 1 ]''')/100
        s[i]    = s[i-1]+s[i-1]*sigma[i]*np.sqrt(dt)*(np.random.randn())
    return [s,sigma]

#def gbm_rough(s0,T,N,h,seed=457778,gamma=1,alpha=1,theta=1,rho=0.0,r=0):
#    ''' corretta''' #giusto
#    S=np.zeros(N)
#    sigma=np.ones(N)
#    y=np.ones(N)
#    y[0]=0
#    sigma[0]=1/100
#    S[0]=s0
#    np.random.seed(seed)
#    dt=T/N
#    cov=np.matrix([[1, rho],[rho, 1]])
#    a  =np.linalg.cholesky(cov)
#    for i in range(1,len(S)):
#        f = FBM(n=1, hurst=h, length=1, method='cholesky')
#        epsilon1=np.random.randn()
#        epsilon2=f.fgn()*a[1, 0]+ f.fgn() * a[1, 1 ]       
#        S[i] = S[i-1]+S[i-1]*sigma[i-1]*np.sqrt(dt)*epsilon1
#        y[i] = (y[i-1] -gamma*y[i-1]*dt+ theta*np.sqrt(dt)*epsilon2)
#        sigma[i] = np.exp(y[i])/100
#    return [S,sigma]
#
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
    for i in range(0,N):
        l[i]=(log(s[i])-log(s[i-1]))**2
    return l

def realVol(x,step,N): #assume che entri in log quadrato
    y=np.zeros(int(N/step))
    iter=step
    iter2=0
    for i in range(0,len(y)):
        y[i]=sum(x[iter2:iter])
        iter2=iter#-step
        iter =iter+step
    for i in range(0,len(y)):
        y[i]=np.sqrt(y[i])*np.sqrt(int(N/step))
    return y[1:]

def W(s,T,K,p):
    W    =np.zeros(np.int64(T/K)+1)
    num  =np.zeros(np.int64(T/K)+1)
    den  =np.zeros(np.int64(T/K)+1)
    diff =np.zeros(len(s)-2)
    iter1=0
    iter2=K
    for j in range(1,len(diff)):
        diff[j-1]= s[j]-s[j-1]
    for i in range(1,len(W)):
        num[i] =np.abs(s[iter2]-s[iter1])**p
        den[i] =sum(np.abs(diff[iter1:iter2])**p)*K
        W[i]   =num[i]/den[i]
        iter1=iter2
        iter2+=K
    return sum(W)


def W_1(s,T,K,p):
    W    =np.zeros(np.int64(T/K)+1)
    num  =np.zeros(np.int64(T/K)+1)
    den  =np.zeros(np.int64(T/K)+1)
    diff =np.zeros(len(s))
    iter1=0
    iter2=K
    for j in range(1,len(diff)):
        diff[j-1]= log(s[j])-log(s[j-1])
    for i in range(1,len(W)):
        num[i] =np.abs(log(s[iter2])-log(s[iter1]))**p
        den[i] =sum(np.abs(diff[iter1:iter2])**p)*K
        W[i]   =num[i]/den[i]
        iter1=iter2
        iter2+=K
        if iter1==(len(diff)-1):
            return sum(W)
    return sum(W)


def calcW(f,K,p):
    diff=np.zeros(len(f))
    w   =np.zeros(int(len(f)/K)+1)
    num =np.zeros(int(len(f)/K)+1)
    den =np.zeros(int(len(f)/K)+1)
    iter=K
    iter2=0
    for j in range(1,len(w)):
        diff[j-1]=abs(log(f[j])-log(f[j-1]))
    for i in range(len(w)):
        num[i]=abs(log(f[iter])-log(f[iter2]))**p *(K)#
        den[i]=sum(diff[iter2:iter])**p#*#(iter-iter2)
        w[i]  = den[i]/num[i]#(num[i]/den[i])
        iter2=iter
        iter+=K
        if iter2>i*K:
            return sum(w)
        #if (iter>=len(w)):
        #    iter=iter-(iter-len(w))-1
        #elif (iter>=i*K):        
        #    return sum(w)                                                                        
        #if iter2>=(len(diff)-1):
        #    return sum(w)
    return sum(w)




def flatten(xss):
    return [x for xs in xss for x in xs]
'''
    def W(s,T,K,p):
    W    =np.zeros(len(s))#np.int64(T/K))
    num  =np.zeros(len(s))#np.zeros(np.int64(T/K))
    den  =np.zeros(len(s))#np.zeros(np.int64(T/K))
    diff =np.zeros(len(s))
    iter1=0
    iter2=K
    for j in range(0,len(diff)):
        diff[j]= s[j]-s[j-1]
    for i in range(0,len(s)):
        num[i] =np.abs(s[iter2]-s[iter1])**p
        den[i] =sum(np.abs(diff[iter1:iter2])**p)
        W[i]   =num[i]/den[i]
        iter1=iter2
        iter2+=K
    return sum(W)
'''