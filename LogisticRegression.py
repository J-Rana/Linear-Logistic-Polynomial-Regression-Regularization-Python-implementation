"""
In the future, Python will switch to always yielding a real result,
and to force an integer division operation you use the special "//"
integer division operator.  If you want that behavior now, just import that "from the future"""
from __future__ import print_function, division

import numpy as np
import math




#Converts vector x to a diagonal matrix
def diagonal_matrix(x):
    
   
    m = np.zeroes((len(x),len(x)))
    for i in range(len(m[0])):
        m[i,i] = x[i]
    return m



class SigmoidFunction():
    def _call_(self, x):
        
        #sigmoid function = 1 / (1 + e^-x)
        return 1/(1 + np.exp(-x))
        
    def gradient(self, x):
        #gradient of sigmoid function = (sigmoid function) * (1 - sigmoid function)
        return self._call_(x) * (1 - self._call_(x))
        
    

#Logistic Regression classifier
class LogisticRegression():
    """Required Parameters:
        learning_rate (float):
            The step length that will be used when updating the weights.
        gradient_descent (boolean) :
            True or false depending on if gradient descent should be used when training. If 
            false then we use batch optimization by least squares.
        """
        
    def _init_(self, learning_rate = .1, gradient_descent = True):
        self.parameters = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = SigmoidFunction()
        
    def parameters_initialization(self, X):
        features = np.shape(X)[1]
        
        #initializing weight randomly, uniformly and close to zero ie, [-1/N, 1/N]
        limit = 1/math.sqrt(features)
        
        
        #uniformly distributes no. of probabilities equal to the size argument between low and high values
        self.parameters = np.random.uniform(-limit, limit, (features))
    
    def fit(self, X, y, iterations = 4000):
        self.parameters_initialization(X)
        
        
        #Do gradient descent for given no. of iterations
        for i in range(iterations):
            #y prediction = 1 / ( 1 + e^-(wx) )
            y_prediction = self.sigmoid(X.dot(self.parameters))
            
            if self.gradient_descent:
                #Move against the gradient of the loss function with respect to the parameters to minimize the loss
                
                # w -= learning rate *  (- ( training y - predicted y ) * training X )
                self.parameters -= self.learning_rate * -(y - y_prediction).dot(X)
                
            else:
                
                #construct a diagonal matrix of sigmoid gradient column vector using "diagonal_matrix" 
                #method and "gradient" method of "SigmoidFunction"  class
                diagonal_sigmoid_gradient =  diagonal_matrix(self.sigmoid.gradient(X.dot(self.parametrs)))
                
                #Batch optimization method: ie, updating the parametrs at once rather than incremental updates
                self.parameters = np.linalg.pinv(X.T.dot(diagonal_sigmoid_gradient).dot(X.T)).dot(diagonal_sigmoid_gradient.dot(X).dot(self.parameters) + (y - y_prediction))
                
                
            def prediction(self, X):
                y_prediction = np.round(self.sigmoid(X.dot(self.parameters))).asType(int)
                return y_prediction
    
        
        
        
    
    