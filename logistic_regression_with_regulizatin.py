import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

""" it is worth noting that the only difference 
between a logisrtic regression and linrear regrression 
is the activation energy. One can think of a
linear rregression with an a linear activation """

def split(data,split_percent=0.7,seed=42):
    random.seed(seed)
    random.shuffle(data)
    m,n=data.shape
    m_train=int(split_percent*m)
    train_data=data[:m_train,:]
    test_data=data[m_train:,:]
    x_train=train_data[:,:n-1]
    y_train=train_data[:,n-1:]
    x_test=test_data[:,:n-1]
    y_test=test_data[:,n-1:]

    return ((x_train,y_train),(x_test,y_test))


def sigmoid_activation(z):
    return 1/(1+np.exp(-z))

def F1Score(y,y_pred):
    assert len(y_pred)==len(y)
    y_pred=np.where(y_pred>.5,1,0)
    accurancy=np.sum(y_pred==y)/len(y_pred)
    return accurancy


def normalization(x):
    norm=(x-np.mean(x))/np.std(x)
    return norm


def cost(y,y_pred,w,m,reg):
    cost=-((np.dot(y,np.log(y_pred).T)+np.dot((1-y),np.log(1-y_pred).T))-(reg/2)*np.sum(reg*w**2))/m
    return cost

def linear_activation(x,w,b):
    z=np.dot(w.T,x)+b
    return z
 
def forward_prop(x,w,b):
    z= linear_activation(x,w,b)
    a= sigmoid_activation(z)
    return a


def parameter_initialization(n):
    w=np.zeros((n,1))*.001
    b=0
    parameter={"w":w,"b":b}
    return parameter

def backprog(x,y,activation,m,reg,w):
    dz=activation-y
    dw=(np.dot(x,dz.T)+reg*w)/m
    db=sum(dz.T)/m
    grad={"dw":dw,"db":db}
    return grad
    

def logistic_regresion(x_train,y_train,x_test,y_test,learning_rate=0.05,reg=0,epoch=50000):
    assert type(epoch)==int
    m,n=x_train.shape
    x_train,y_train=x_train.T,y_train.T
    parameter=parameter_initialization(n)
    w=parameter["w"]
    b=parameter["b"]
    y=y_train
    
    x=normalization(x_train)
    q1=range(0,epoch,int(epoch/20)+1)
    co=[]
    epo=[]
    best_cost=float("inf")
    for i in range(epoch):
        
        activation=forward_prop(x,w,b)
        grad=backprog(x,y,activation,m,reg,w)
        w-=learning_rate*grad["dw"]
        b-=learning_rate*grad["db"]
        y_pred=activation
        cost_train=cost(y,y_pred,w,m,reg)
        if i in q1:
            epo.append(i)
            co.append(cost_train[0,0])
        if cost_train< best_cost:
            best_cost=cost_train
            patience=0
        else:
            patience+=1
        if patience>=5:
            break
    print("number of iteration:",i)
    plt.plot(epo,co)
    print("cost of train data=",cost_train)
    x=normalization(x_test.T)
    y=y_test.T
    y_pred_test=forward_prop(x,w,b)
    cost_test=cost(y,y_pred_test,w,m,reg)
    print("cost of test data=",cost_test)
    parameter= {"w":w,"b":b}
    return parameter



# from sklearn.datasets import load_breast_cancer
# x,y=load_breast_cancer(return_X_y=True)
# data=np.append(x,y.reshape(len(y),1),axis=1)
# (x_train,y_train),(x_test,y_test)=split(data,.5)
# logistic_regresion(x_train,y_train,x_test,y_test,.06,reg=10)







plt.show()

