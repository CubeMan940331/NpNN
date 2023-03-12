import numpy as np
import os
import platform
def fs(x): return 1.0/(1+np.exp(-x))
def activation(x,w,b): return fs(np.dot(x,w)+b)
def echo_ans(y): return np.argmax(y)
class net:
    def init_(self):
        self.L=len(self.shape_list)-1
        self.w_list=[]
        self.b_list=[]
        self.a_list=[i for i in range(self.L+1)]
        for i in range(self.L):
            weight=np.random.randn(self.shape_list[i],self.shape_list[i+1])
            bias=np.ones(self.shape_list[i+1])/10.0
            self.w_list.append(weight)
            self.b_list.append(bias)
    #===========================================
    def load(self,dir_name):
        if platform.system()=='Windows':
            path='.\\'+dir_name+'\\'
        else:  path='./'+dir_name+'/'
        self.shape_list=list(np.load(path+'shape.npy'))
        self.L=len(self.shape_list)-1
        self.a_list=[i for i in range(self.L+1)]
        self.w_list=[]
        self.b_list=[]
        for i in range(self.L):
            self.w_list.append(np.load(path+'w'+str(i)+'.npy'))
            self.b_list.append(np.load(path+'b'+str(i)+'.npy'))
    #===========================================
    def __init__(self,init_config):
        if type(init_config)==list:
            self.shape_list=init_config
            self.init_()
        else: self.load(init_config)
    #===========================================
    def calculate(self,x):
        self.a_list[0]=x
        for i in range(self.L):
            self.a_list[i+1]=activation(self.a_list[i],self.w_list[i],self.b_list[i])
    #===========================================
    def train(self,x,y,eta):
        self.calculate(x)
        p_d=self.a_list[-1]-y
        p_d=p_d/x.shape[0]
        #dC/da(L)
        for l in range(self.L,0,-1):
            p_d*=self.a_list[l]*(1-self.a_list[l])
            #####da(l)/dz(l)
            dzdw=self.a_list[l-1].T#dz(l)/dw(l-1)=a(l-1)T
            dzda=self.w_list[l-1].T#dz(l)/da(l-1)=w(l-1)T
            ############
            self.b_list[l-1]-=eta*np.sum(p_d,axis=0)
            self.w_list[l-1]-=eta*np.dot(dzdw,p_d)                  
            #########
            p_d=np.dot(p_d,dzda)
    #===========================================
    def cost(self,x=None,y=None):
        if x is not None and y is not None:
            self.calculate(x)
        return np.sum((self.a_list[-1]-y)**2)/y.shape[0]
    #===========================================
    def save(self,dir_name:str,protect=True):
        if platform.system()=='Windows':
            path='.\\'+dir_name+'\\'
        else: path='./'+dir_name+'/'
        os.makedirs(path,exist_ok=(not protect))
        tmp=np.array(self.shape_list)
        np.save(path+'shape',tmp)
        for i in range(self.L):
            np.save(path+'b'+str(i),self.b_list[i])
            np.save(path+'w'+str(i),self.w_list[i])
    #===========================================
    def accuracy(self,x=None,y=None):
        correct_count=0
        if x is not None and  y is not None:
            self.calculate(x)
        for i in range(self.a_list[-1].shape[0]):
            if echo_ans(self.a_list[-1][i])==echo_ans(y[i]):
                correct_count+=1
        return correct_count/self.a_list[-1].shape[0]
    