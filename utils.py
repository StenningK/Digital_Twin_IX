import numpy as np
import matplotlib.pyplot as plt
import torch

class ESN:
    
    def __init__(self,N,N_in,N_av,alpha,rho,gamma):        
        
        self.N=N
        self.alpha=alpha
        self.rho=rho
        self.N_av=N_av
        self.N_in=N_in
        self.gamma=gamma
        
        diluition=1-N_av/N
        W=np.random.uniform(-1,1,[N,N])
        W=W*(np.random.uniform(0,1,[N,N])>diluition)
        eig=np.linalg.eigvals(W)
        self.W=torch.from_numpy(self.rho*W/(np.max(np.absolute(eig)))).float()
        
        
        self.x=[]
        
        if self.N_in==1:
            
            self.W_in=2*np.random.randint(0,2,[self.N_in,self.N])-1
            self.W_in=torch.from_numpy(self.W_in*self.gamma).float()
            
            
        else:
            
            self.W_in=np.random.randn(self.N_in,self.N)
            self.W_in=torch.from_numpy(self.gamma*self.W_in).float()
        
        
    def Reset(self,s):
        
        batch_size=np.shape(s)[0]
        self.x=torch.zeros([batch_size,self.N])
        
    def ESN_1step(self,s):
        
        self.x=(1-self.alpha)*self.x+self.alpha*torch.tanh(torch.matmul(s,self.W_in)+torch.matmul(self.x,self.W))
        
    def ESN_response(self,Input):
        
        T=Input.shape[2]
        X=torch.zeros(Input.shape[0],self.N,T)
        
        self.Reset(Input[:,0])
        
        for t in range(T):
            
            self.ESN_1step(Input[:,:,t])
            X[:,:,t]=torch.clone(self.x)
            
        return X
class Physical_Device:

    def __init__(self,N,N_in,N_av,N_out,alpha,rho,gamma,noise=0.01):

        self.N=N
        self.alpha=alpha
        self.rho=rho
        self.N_av=N_av
        self.N_in=N_in
        self.gamma=gamma
        self.N_out = N_out
        self.noise=noise
        diluition=1-N_av/N
        W=np.random.uniform(-1,1,[N,N])
        W=W*(np.random.uniform(0,1,[N,N])>diluition)
        eig=np.linalg.eigvals(W)
        self.W=torch.from_numpy(self.rho*W/(np.max(np.absolute(eig))))#.float()

        out_channels = []

        for i in range(self.N_out):
          new=False
          while new == False:
            channel = np.random.randint(0,N)
            if channel not in out_channels:
              out_channels.append(channel)
              new=True

        self.out_channels = out_channels
        self.x=[]

        if self.N_in==1:

            self.W_in=2*np.random.randint(0,2,[self.N_in,self.N])-1
            self.W_in=torch.from_numpy(self.W_in*self.gamma)#.float()


        else:

            self.W_in=np.random.randn(self.N_in,self.N)
            self.W_in=torch.from_numpy(self.gamma*self.W_in)#.float()


    def Reset(self,s):

        batch_size=np.shape(s)[0]
        self.x=torch.zeros([batch_size,self.N])

    def single_point(self,s):

        self.x=(1-self.alpha)*self.x+self.alpha*torch.tanh(torch.matmul(s,self.W_in)+torch.matmul(self.x,self.W))+torch.normal(0.0,self.noise,size=(1,1))

    def run_sequence(self,Input):

        T=Input.shape[2]
        X=torch.zeros(Input.shape[0],self.N,T)

        self.Reset(Input[:,0])

        for t in range(T):

            self.single_point(Input[:,:,t])
            X[:,:,t]=torch.clone(self.x)

        #print(X.shape)
        #print(self.out_channels)
        return X[:,self.out_channels]