# DNN model

#import necessary libraries
import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from random import shuffle
from testCases_v2 import *
from dnn_app_utils_v2 import *

# load the dataset and the labels
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test =  np.load('x_test.npy')
y_test = np.load('y_test.npy')

print(x_train.shape)
print(y_train.shape)
print('============')
print(x_test.shape)
print(y_test.shape)

# normalize the inputs

x_train = x_train / 255
x_test = x_test / 255



# create a three different layers
layers_dims1 = [x_train.shape[0], 20, 7, 5, 1] #3 hidden layers
layers_dims2 = [576,4,1]#single hidden layer
layers_dims3 = [576,45,20,7,6,4,1]#5 hidden layers

# create the model for the dnn model
def L_layer_model(X, Y, layers_dims, learning_rate = 0.1, num_iterations = 3000, print_cost=False):#lr was 0.009
    

    #fix the random initialization
    np.random.seed(1)
    # keep track of cost
    costs = []                         
    
    # Parameters initialization.
   
    parameters = initialize_parameters_deep(layers_dims)
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
         
        AL, caches = L_model_forward(X, parameters)
        
        
        # Compute cost.
        
        cost = compute_cost(AL, Y)
        
    
        # Backward propagation.
        
        grads = L_model_backward(AL, Y, caches)
        
 
        # Update parameters.
        
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
        
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        #if print_cost and i % 100 == 0:
        costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

#3 hidden layers
parameters = L_layer_model(x_train, y_train, layers_dims1, num_iterations = 9877, print_cost = True)



### train accuracy

# comparing the predicted value and the true vale
def prediction(x,parameters):
    
    AL, caches = L_model_forward(x, parameters)
    y1 = ( AL > 0.5 )
    return y1
# create a function to calculate the accuracy
def accuracy(y1,y):
    return 100 - (np.mean(np.abs(np.subtract(y,y1),y)) * 100)
prediction(x_train,parameters)
accuracy(prediction(x_train,parameters),y_train)


### test accuracy

prediction(x_test,y_test,parameters)

w1 = parameters['W1']
w2 = parameters['W2']
w3 = parameters['W3']
w4 = parameters['W4']
b1 = parameters['b1']
b2 = parameters['b2']
b3 = parameters['b3']
b4 = parameters['b4']

weights = {'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4}

biases = {'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4}


#save the weights and the biases
np.save('weights.npy',weights)
np.save('biases.npy',biases)

parameters = L_layer_model(x_train, y_train, layers_dims3, num_iterations = 9877, print_cost = True,learning_rate=0.075)

