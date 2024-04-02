# Imports
import torch

# Linear model class definition

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return torch.matmul(X, self.w) 

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """

        s = self.score(X)
        return 1.0*(s > 0)
    

# Perceptron class definition

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}        
        """

        y_hat = 2 * self.predict(X) - 1
        missclass = 1.0*(y_hat * y < 0)
        return missclass.mean()

    def grad(self, X, y):
        """
        Computes the vector to add to w. 

        ARGUMENTS: 
            X, torch.Tensor: the observation of the feature matrix used in the current update.

            y, torch.Tensor: the element of the target vector used in the current update.                
        """
        s = X@self.w
        return (s*y < 0)*X*y
         

# PerceptronOptimizer class definition

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        # Choose random int i
        n = X.size()[0]
        i = torch.randint(n, size = (1,))

        # Subset X and y based on i
        x_i = X[[i],:]
        y_i = y[i]

        # Record loss before change
        current_loss = self.model.loss(X, y)

        # Update perceptron, adding result to w
        self.model.w += torch.reshape(self.model.grad(x_i, y_i),(self.model.w.size()[0],))
        
        # Record loss after change
        new_loss = self.model.loss(X, y)

        # Return values for  visualizations
        return i, abs(current_loss - new_loss)
        