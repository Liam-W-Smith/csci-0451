# import packages
import torch

# define linear regression class
class LinearRegress:

    def __init__(self, penalty = None, lam = 0):
        self.w = None 
        self.lam = lam
        self.penalty = penalty

    def pred(self, X):
        """
        Compute predictions for each observation in X. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

        RETURNS: 
            torch.Tensor: vector of scores
        """
        return X@self.w
    
    def mse(self, X, y):
        """
        Compute the mean squared error for the model given a set of predictors and outcomes. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the target vector.  y.size() = (n,).

        RETURNS: 
            torch.Tensor: the mean squared error
        """
        return ((self.pred(X) - y)**2).mean()
    
    def loss(self, X, y):
        """
        Compute the loss for the model, accounting for any penalty terms. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the target vector.  y.size() = (n,).

        RETURNS: 
            torch.Tensor: the loss for the current model
        """
        norm = 0
        if self.penalty == "l1":
            norm = torch.sum(torch.abs(self.w[:-1]))
        if self.penalty == "l2":
            norm = torch.sum(self.w[:-1]**2)
        return self.mse(X, y) + self.lam * norm
    
    def grad(self, X, y):
        """
        Compute the gradient of the loss function. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the target vector.  y.size() = (n,).

        RETURNS: 
            torch.Tensor: the gradient of the loss function
        """
        n = X.size()[1]
        norm_grad = 0
        if self.penalty == "l1":
            norm_grad = torch.sign(self.w)
            norm_grad[-1] = 0
        if self.penalty == "l2":
            norm_grad = 2*self.w
            norm_grad[-1] = 0
        return (2/n)*(X.transpose(0, 1))@(X@self.w-y) + self.lam*norm_grad
    
    def r2(self, X, y):
        """
        Compute the coefficient of determination (R-squared) for the model given a set of predictors and outcomes. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features.

            y, torch.Tensor: the target vector.  y.size() = (n,).

        RETURNS: 
            torch.Tensor: the coefficient of determination (R-squared)
        """
        y_hat = self.pred(X)
        y_bar = torch.mean(y)
        return 1 - torch.sum((y - y_hat)**2)/torch.sum((y - y_bar)**2)
    
# define gradient descent optimizer class
class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, alpha):
        """
        Compute one iteration of gradient descent using the feature matrix X 
        and target vector y. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix.

            y, torch.Tensor: the target vector.

            alpha, float: the learning rate for gradient descent.
        """

        # Initialize w if needed
        if self.model.w is None: 
            self.model.w = torch.rand((X.size()[1]), dtype = torch.float64)

        # Update w
        self.model.w -= alpha*self.model.grad(X, y)