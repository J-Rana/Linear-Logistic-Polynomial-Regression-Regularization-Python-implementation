
import numpy as np
import math



def features_polynomial(X, degree):
    samples, features = np.shape(X)
    
 
    
#Normalize data set X by dividing it with L2 norm
def normalize_rescale(X, axis = -1, order =2):
    l2_norm = np.atleast_1d(np.linalg.norm(X, order, axis))
    #Calculating the size or length of a vector = norm  
    
    #L2 norm is square root of sum of squared elements of a vector 
    l2_norm[l2_norm == 0]=1
        
    #dividing data set with L2 norm of each of its vector to normalize
    return X / np.expand_dims(l2_norm, axis)
   
    


    
class lasso_l1_regularization():
    #Lasso regularization: adds absolute weight magnitude to the loss function
    
    
    #The __init__ method is roughly what represents a constructor in Python.
    def _init_(self, alpha):
        self.alpha = alpha
     
        
   
    def _call_(self, w):
        #multiplys weights with alpha
        return self.alpha * np.linalg.norm(w)
    
    
    #multiplying alpha with sign of each elemement in the weight vector w and returning the array.
    def gradient(self, w):
        return self.alpha * np.sign(w)
    
        
   
    
    
class ridge_l2_regularization():
    #Ridge regularization adds squared weight to the loss function
    
    
    def _init_(self, alpha):
        self.alpha = alpha
    
    
    #alpha * 1/2 * w^2 = ridge regularization term
    def _call_(self, w):
        return self.alpha * 0.5 * w.T.dot(w)
       
    
    def gradient(self, w):
        return self.alpha * w 
        #sign of alpha not required because already squared
   

     
        
        
 #alpha * |w| + (1-alpha) * |w|^2 = penalty ters contributes by l1 and l2 respectively.
class elastic_net_regularization():
    #elastic net regularization linearly combines L1 and L2 ie, lasso and ridge regression penalties.
    
    
    def _init_(self, alpha, l1_ratio=0.5):
        self.lpha = alpha
        self.l1_ratio = l1_ratio
        
        
    def _call_(self, w):
        l1_contribution_term = self.l1_ratio * np.linalg.norm(w)
        l2_contribution_term = (1-self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contribution_term +l2_contribution_term  )
    
    
    def gradient(self, w):
        l1_contribution_term = self.l1_ratio * np.sign(w)
        l2_contribution_term = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contribution_term  + l2_contribution_term ) 
    
    
    
    
class Regression(object):
#class inherits the base class called "object"
   
    """
    Regression models the relationship between input X and output Y.
    Required parameters:
    iterations(float) = number of iterations required by algorithm to find weights w using training data
    learning_rate (float) = length of the step required to update the weights w.
    
    """
    def _init_(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate
        
        
    def weights_initialization(self, features):
        #initializing weight randomly, uniformly and close to zero ie, [-1/N, 1/N]
        limit = 1/math.sqrt(features)
        self.w = np.random.uniform(-limit, limit, (features))
        
    
    def fit(self, X, y):
        #Adding a bias weight that does not depend on any of the features allows the hyperplane desbribed by your 
        #learned weights to more easily fit data that doesn't pass through the origin.
        """A bias unit is an "extra" neuron added to each pre-output layer that stores 
        the value of 1. Bias units aren't connected to any previous layer 
        and in this sense don't represent a true "activity"."""
        
        #insert constant ones for bias weights 
        X = np.insert(X,0,1,axis =1)
        
        #initializing training error array
        self.training_errors = []
        
        #intializing weights
        self.weights_initialization(features = X.shape[1])
        
        
        #Do gradient descent for given no. of iterations
        for i in range(self.iterations):
            y_prediction = X.dot(self.w)
            
            #calculate l2 ridege loss
            #mse = mean squared error
            mse = np.mean(0.5 * (y - y_prediction)**2 + self.regularization(self.w))
            
            #store mean squared error in training errors array
            self.training_errors.append(mse)
            
            
            #Gradient of l2 ridge loss w.r.t to weight w (by differentiating)
            gradient_w = -(y - y_prediction).dot(X) + self.regularization.gradient(self.w)
            
            #update weights
            self.w -= self.learning_rate * gradient_w
           
            
    #predict the y value        
    def prediction(self, X):
        #inserting constant ones for bias weights
        X= np.insert(X, 0, 1, axis = 1)
        
        #y = X * w
        y_prediction = X.dot(self.w)
        return y_prediction
    
    
    
    
    
class LinearRegression(Regression):
    #here Regression class is the argument of LinearRegression class
    """Linear Model
    Required Parameters:
    #iterations (float): 
        number of iterations required by algorithm to find weights w using training data
    #learning_rate (float): 
        The step length that will be used when updating the weights.
     #gradient_descent (boolean)  :
         True or false depending on if gradient descent should be used when training. If 
         false then we use batch optimization by least squares."""
         
    
    def _init_(self, iterations= 100, learning_rate = 0.001, gradient_descent = True):
        self.gradient_descent = gradient_descent
            
        #No regularization
        self.regularization = lambda x : 0
        #lambda arguments : expression
        self.regularization.gradient = lambda x : 0
        super(LinearRegression, self)._init_(iterations = iterations, learning_rate = learning_rate)
        
        
    def fit(self, X, y):
        
        # If not gradient descent => Least squares approximation of w ie, batch optimization
        if not self.gradient_descent:
            
            #insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis = 1)
            
            #Calculate weights by least square method using (using Moore-Penrose pseudoinverse)
            """When inverse does not exist we find pseudo inverse using matrices U, S, V where S 
            is the diagonal matrix and U, V are unitary matrices
            SVD: Single value decomposition:
            decomposes the matrix X effectively into rotations U and V and the diagonal matrix S."""
            U, S, V= np.linalg.svd(X.T.dot(X))
            
            #np.diag(S): creates an array out of diagonal elements of the matrix
            S = np.diag(S)
            
            """.T:
                The .T accesses the attribute T of the object, which happens to be a NumPy array. 
                The T attribute is the transpose of the array. 
                
            pinv:  Compute the (Moore-Penrose) pseudo-inverse of a matrix.  
                Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD)
                and including all large singular values.
                
                """
            
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            
            #w= ((X^T * X)^-1) * (X^T) * y
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
            
        
        else:
            #if gradient descent is required we do the regression
            super(LinearRegression, self).fit(X,y)
            
            
            
            
            
            
class LassoRegression(Regression):
    """ Lasso regression uses regularization model with the loss function to best fit the model w.r.t training data
    and complexity of the model.
    Required Parameters:
        degree_X (int):
            degree of polynomial of independent input variable X
        regularization_factor (float):
            Determines the regularization factor ie, the amount of feature shrinkage or extraction needed.
         iterations (float):
             number of iterations required by algorithm to find weights w using training data
         learning_rate (float):
             The step length that will be used when updating the weights.   
             """
           
            
    def _init_(self, degree_X, regularization_factor , iterations= 3000, learning_rate = 0.01):
        self.degree_X = degree_X
        
        #using lasso  class "lasso_l1_regularization" to calculate regularization 
        self.regularization = lasso_l1_regularization(alpha = regularization_factor)
        
        #Using "Regression" class here with regularization of "LassoRegression" class
        super(LassoRegression, self)._init_(iterations, learning_rate)
        
    
    def fit(self, X, y):
        #using methods "normalize_rescale" and "features_polynomial" to normalize the data set
        X = normalize_rescale(features_polynomial(X, degree = self.degree))
        
        #using fit method of "Regression" class
        super(LassoRegression, self).fit(X,y)
        
        
    def prediction(self, X):
        #normalizing data set unlike we did in "Regression" class's "prediction" method
        X = normalize_rescale(features_polynomial(X, degree = self.degree))
        return super(LassoRegression, self).prediction(X)
        
    
    
      



class PolynomialRegression(Regression):
    """Polynomial Regression uses non-linear model to best fit the data
    Required Parameters:
        degree (int):
            degree of polynomial of independent input variable X
        iterations (float):
            number of iterations required by algorithm to find weights w using training data
        learning_rate (float) :  
            The step length that will be used when updating the weights.   
        """
        
        
    def _init_(self, degree, iterations = 3000, learning_rate = 0.001)   :
        self.degree = degree
        
        #No regulareization
        self.regularization = lambda x : 0
        self.regularization.gradient = lambda x : 0
        super(PolynomialRegression, self)._init_(iterations=iterations, learning_rate= learning_rate)
        
    
    def fit(self, X, y):
        #using "features_polynomial" method
        X = features_polynomial(X, degree = self.degree)
        
        #using fit method of "Regression" class
        return super(PolynomialRegression, self).fit(X, y)


    def prediction(self, X):
        ##using "features_polynomial" method
        X = features_polynomial(X, degree = self.degree)
        #using prediction method of "Regression" class
        return super(PolynomialRegression, self).prediction(X)







class RidgeRegression(Regression):   
    """Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.
    Parameters:
  
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    
            
    def _init_(self, degree, regularization_factor, iterations =3000, learning_rate = 0.01, gradient_descent = True):
        self.degree = degree
        
        #use "ridge_l2_regularization" class
        self.regularization = ridge_l2_regularization(alpha = regularization_factor)
        super(RidgeRegression, self)._init_(iterations, learning_rate)
        
    
        
    
    
       
class PolynomialRidgeRegression(Regression):
    
    
     """Similar to regular ridge regression except that the data is transformed to allow
    for polynomial regression.
    Required Parameters:
        degree (int):
            degree of polynomial of independent input variable X
        iterations (float):
            number of iterations required by algorithm to find weights w using training data
        learning rate (float) :  
            The step length that will be used when updating the weights.   
        regularization_factor (float):
            The factor that will determine the amount of regularization and feature
            shrinkage. """
    
    
        def _init_(self, degree, regularization_factor, iterations =3000, learning_rate = 0.01, gradient_descent = True):
            self.degree = degree
            
            #use "ridge_l2_regularization" class
            self.regularization = ridge_l2_regularization(alpha = regularization_factor)
            
            super(PolynomialRidgeRegression, self)._init_(iterations, learning_rate)
        
        
        def fit(self, X, y):
            #using methods "normalize_rescale" and "features_polynomial" to normalize the data set
            X = normalize_rescale(features_polynomial(X, degree = self.degree))
            super(PolynomialRidgeRegression, self).fit(X, y)
            
         
             
  
    
    
class ElasticNetRegression(Regression):     
    """Regression where a combination of l1 and l2 regularization are used. The
ratio of their contributions are set with the 'l1_ratio' parameter.
Required Parameters:
degree: int
    The degree of the polynomial that the independent variable X will be transformed to.
regularization_factor: float
    The factor that will determine the amount of regularization and feature
    shrinkage. 
l1_ration: float
    Weighs the contribution of l1 and l2 regularization.
iterations: float
    The number of training iterations the algorithm will tune the weights for.
    """
    
    
    def _init_(self, degree = 1, regularization_factor = 0.05, li_ratio = 0.5, iterations = 3000, learning rate = 0.01):
        self.degree = degree
        
        #using "elastic_net_regularization" class
        self.regularization = elastic_net_regularization(alpha = regularization_factor, l1_ratio = l1_ratio)
        super(ElasticNetRegularization, self)._init_(iterations, learning_rate)
        
        
    def fit(self, X, y):
        #using methods "normalize_rescale" and "features_polynomial" to normalize the data set
        X = normalize_rescale(features_polynomial(X, degree = self.degree))
        super(ElasticNetRegularization, self).fit(X,y)
        
        
    def prediction(self, X):
        #using methods "normalize_rescale" and "features_polynomial" to normalize the data set
        X = normalize_rescale(features_polynomial(X, degree = self.degree))
        return super(ElasticNetRegularization, self).prediction(X)
    
    
    

        
        
    
   
    
        
    
    

    



        
    
    



    

    