# Import packages
import torch

# Linear model class definition

class LinearModel:

    def __init__(self):
        self.w_old = None
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
            s, torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return torch.matmul(X, self.w) 

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X.
        The prediction for the ith data point is either 0 or 1. 

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

class LogisticRegression(LinearModel):

    def loss(self, X, y):
        """
        Computes the binary cross-entropy loss function used in logistic regression.

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1} 

        RETURNS: 
            l, torch.Tensor: the binary cross-entropy loss for the current model       
        """
        n = X.size()[0]
        s = X@self.w
        return (1/n)*torch.sum(-y*torch.log(1/(1+torch.exp(-s))) - (1 - y)*torch.log(1 - 1/(1+torch.exp(-s))))


    def grad(self, X, y):
        """
        Computes the gradient of the binary cross-entropy loss function.

        ARGUMENTS: 
            X, torch.Tensor: the observation of the feature matrix used in the current update.

            y, torch.Tensor: the element of the target vector used in the current update.  

        RETURNS: 
            g, torch.Tensor: the gradient of our loss function              
        """
        n = X.size()[0]
        s = X@self.w
        v = 1/(1+torch.exp(-s)) - y
        v_ = v[:, None]
        return (1/n)*(v_*X).sum(axis = 0)
         

# PerceptronOptimizer class definition

class AdamOptimizer:

    def __init__(self, model, batch_size, alpha, beta_1, beta_2, w_0 = None):
        self.model = model 
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.w = w_0
    
    def step(self, X, y):
        """
        Compute one iteration of gradient descent with momentum using the feature matrix X 
        and target vector y. 

        ARGUMENTS: 
            X, torch.Tensor: the observation of the feature matrix used in the current update.

            y, torch.Tensor: the element of the target vector used in the current update.  

            alpha, float: the learning rate for regular gradient descent

            beta, float: an additional learning rate for gradient descent with momentum
        """

        # Initialize w if needed
        if self.model.w is None: 
            self.model.w = torch.rand((X.size()[1]))
        if self.model.w_old is None: 
            self.model.w_old = torch.rand((X.size()[1]))

        # Call the loss function
        self.model.loss(X, y)

        # Record current w 
        temp = self.model.w

        # Update w
        self.model.w += -alpha*self.model.grad(X, y) + beta*(self.model.w - self.model.w_old)

        # Update old w
        self.model.w_old = temp
        