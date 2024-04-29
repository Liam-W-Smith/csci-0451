# Import packages
import torch

# Sparse Kernel Logistic Regression class definition

class KernelLogisticRegression():

    def __init__(self, X, kernel, lam, gamma):
        self.a = None
        self.Xt = X
        self.k = kernel
        self.lam = lam
        self.gamma = gamma


    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 

        If self.a currently has value None, then it is necessary to first initialize self.a to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.
 

        RETURNS: 
            s, torch.Tensor: vector of scores. s.size() = (n,)
        """

        return (self.k(self.Xt, X, self.gamma).transpose(0, 1))@self.a

    # def predict(self, X):
    #     """
    #     Compute the predictions for each data point in the feature matrix X.
    #     The prediction for the ith data point is either 0 or 1. 

    #     ARGUMENTS: 
    #         X, torch.Tensor: the feature matrix. X.size() == (n, p), 
    #         where n is the number of data points and p is the 
    #         number of features. This implementation always assumes 
    #         that the final column of X is a constant column of 1s. 

    #     RETURNS: 
    #         y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
    #     """

    #     s = self.score(X)
    #     return 1.0*(s > 0)
    
    def loss(self, X, y):
        """
        Computes the loss function used in sparse kernelized logistic regression.

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1} 

        RETURNS: 
            l, torch.Tensor: the loss for the current model       
        """

        s = self.score(X)
        m = X.size()[0]
        sum = torch.sum(y*torch.log(1/(1+torch.exp(-s))) + (1 - y)*torch.log(1 - 1/(1+torch.exp(-s))))
        reg_norm = self.lam * torch.sum(torch.abs(self.a))

        return (1/m) * sum + reg_norm


    def grad(self, X, y):
        """
        Computes the gradient of the binary cross-entropy loss function.

        ARGUMENTS: 
            X, torch.Tensor: the observation of the feature matrix used in the current update.

            y, torch.Tensor: the element of the target vector used in the current update.  

        RETURNS: 
            g, torch.Tensor: the gradient of our loss function              
        """
        m = X.size()[0]
        s = self.score(X)
        v = 1/(1+torch.exp(-s)) - y
        v_ = v[:, None]
        a_reg_grad = torch.sign(self.a)
        return (-1/m)*(v_*X).sum(axis = 1) + self.lam*a_reg_grad
         

# GradientDescentOptimizer class definition

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, alpha):
        """
        Compute one iteration of gradient descent with momentum using the feature matrix X 
        and target vector y. 

        ARGUMENTS: 
            X, torch.Tensor: the observation of the feature matrix used in the current update.

            y, torch.Tensor: the element of the target vector used in the current update.  

            alpha, float: the learning rate for regular gradient descent

            beta, float: an additional learning rate for gradient descent with momentum
        """

        # Initialize a if needed
        if self.model.a is None: 
            self.model.a = torch.rand((X.size()[0]))

        # # Call the loss function
        # self.model.loss(X, y)

        # # Record current a
        # temp = self.model.a

        # Update a
        self.model.a -= alpha*self.model.grad(X, y)
        