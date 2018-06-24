### eye state model with single layer

import numpy as np 
import glob
import cv2 as cv


#import images

train_path = 'eye/train/'
test_path = 'eye/test'

train_files = glob.glob(train_path + '/*/*.jpg')
test_files = glob.glob(test_path + '/*/*.jpg')
train_files[2]


#prepare labels for the images

#for training
train_y = [0 if 'cat' in files else 1 for files in train_files]
test_y = [0 if 'cat' in files else 1 for files in test_files]
train_y[2]


#convert the images to np arrays
x_train = []
for i in train_files:
    img = cv.imread(i)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    x_train.append(img)
x_test = []
for i in test_files:
    img = cv.imread(i)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    x_test.append(img)
    
x_train = np.array(x_train)
x_test = np.array(x_test)
#convert the labels to np arrays
train_y = np.array(train_y)
test_y = np.array(test_y)
print(x_train.shape)

#preprocess the data

#flatten and reshape
x_train = x_train.reshape(x_train.shape[0],-1).T
x_test = x_test.reshape(x_test.shape[0],-1).T

train_y = train_y.reshape(-1,1).T
test_y = test_y.reshape(-1,1).T

#preprocess
x_train = x_train/255
x_test = x_test /255
print(x_train.shape)
print(x_train)
print(train_y.shape)

#single_layer_network


# layer size
def layer_size(x,y):
    n_x = x.shape[0]
    n_h = 576
    n_y = y.shape[0]
    return n_x,n_h,n_y
layer_size(x_train,train_y)
    

#ini_parameters

def ini_parameters(n_x,n_h,n_y):
    np.random.seed(3)
    w1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros(shape = (n_h,1))
    w2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape = (n_y,1))
    
    assert(w1.shape == (n_h,n_x))
    assert(w2.shape == (n_y,n_h))
    assert(b1.shape == (n_h,1))
    assert(b2.shape ==(n_y,1))
    params = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    return params


#relu
def relu(z):
    return z * (z>0)
#propagate
def propagate(x,params):
    
    w1 = params['w1']
    w2 = params['w2']
    b1 = params['b1']
    b2 = params['b2']
    
    z1 = np.dot(w1,x) + b1 
    a1 = np.tanh(z1)
    '''
    print(w1.shape,x.shape,b1.shape,z1.shape,a1.shape)
    print('====')
    print(w2.shape,a1.shape,b2.shape)
    '''  
    z2 = np.dot(w2,a1) + b2
    a2 = (1/1+np.exp(-z2))
    
    cache = { 'z1':z1,'z2':z2,'a1':a1,'a2':a2}
    return a2,cache
#propagate(x_train,1728,4,1)

#define the cost
def cost(y,cache):
    a2 = cache['a2']
    log = np.add(np.multiply(np.log(a2),y) , np.multiply((1-a2),(1-y)))
    cost = -(np.sum(log)) / y.shape[1]
    return cost

#cost(train_y,x_train,1728,4,1)

#backward propagation
def back_prop(params,cache,x,y):
    w1 = params['w1']
    w2 = params['w2']
    b1 = params['b1']
    b2 = params['b2']
    m = y.shape[1]
    a1 = cache['a1']
    a2 = cache['a2']
    z1 = cache['z1']
    z2 = cache['z2']
    
    gz1 = (1-np.power(a1,2)) 
    
    dz2 = np.subtract(a2,y)
    dw2 = np.dot(dz2,a1.T)/m
    db2 = np.sum(dz2,axis = 1, keepdims = True) / m
    dz1 = np.multiply((np.dot(w2.T,dz2)),gz1)
    dw1 = np.dot(dz1,x.T)/m
    db1 = np.sum(dz1,axis = 1, keepdims = True) / m
    
    grads = {'dw1':dw1, 'db1':db1, 'dw2':dw2, 'db2':db2}
    
    return grads

#update the parameters w and b
def update_parameters(grads,params,alpha):
    
    dw1 = grads['dw1']
    dw2 = grads['dw2']
    db1 = grads['db1']
    db2 = grads['db2']
    
    w1 = params['w1']
    w2 = params['w2']
    b1 = params['b1']
    b2 = params['b2']
    
    
    
        
    w1 = w1 - (alpha * dw1 )
    w2 = w2 - (alpha * dw2 )
    b1 = b1 - (alpha * db1 )
    b2 = b2 - (alpha * db2 )
        
    params = {'w1':w1, 'w2':w2, 'b1':b1, 'b2':b2}
    
    return params
    
    

#the model
def train(x,y,alpha,n_iter):
    #get the layer size
    n_x = layer_size(x,y)[0]
    n_y = layer_size(x,y)[2]
    n_h = 576
    #initialize parameters while propagating
    params = ini_parameters(n_x,n_h,n_y)
    costs = []
    for i in range (0,n_iter):
        
        a2,cache = propagate(x,params)
        
        coste = cost(y,cache)
       
        grads = back_prop(params,cache,x,y)
        
        params = update_parameters(grads,params,alpha)
        
        if i % 100 == 0:
            print("cost after" ,i, "iteration = " ,coste)
    
        costs.append(coste)  
    return params,costs
            

def predict(x,params):
        a2 = propagate(x,params)[0]
        
        predict = np.round(a2)
        print(type(a2))
        return predict, a2


