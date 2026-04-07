import numpy as np
from scipy.optimize import minimize


class MaskedLinearRegression:
    """
    A linear regression model with masking.
    """
    
    def __init__(self):
        """
        Initialize model.
        """


    def fit(self, X, y, mask):
        """
        Fit model.
        """

        def masked_mse_loss(weights, X, y, mask):
            """
            Loss function.
            """

            # reshape weights to match the shape needed for the output
            weights = weights.reshape(X.shape[1], y.shape[1])

            # get predictions
            predictions = np.dot(X, weights)

            # get masked errors
            squared_diff = (predictions - y) ** 2
            masked_squared_diff = squared_diff * mask
            
            return np.mean(masked_squared_diff)
        
        # add a column of ones to X to include the intercept in the model
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # initialize weights
        initial_weights = np.zeros((X.shape[1], y.shape[1])).ravel() # flattened

        # minimize the custom loss
        result = minimize(masked_mse_loss, initial_weights, args = (X, y, mask), method = 'L-BFGS-B')

        # get results
        self.optimized_weights = result.x.reshape(X.shape[1], y.shape[1]) # (256, 11)
        y_pred = np.dot(X, self.optimized_weights)
        mse = np.mean(((y_pred - y) ** 2) * mask)
        r2 = 1 - (np.sum(((y_pred - y) ** 2) * mask) / np.sum(((y - np.mean(y, axis = 0)) ** 2) * mask))

        return y_pred, mse, r2
    

    def predict(self, X):
        """
        Make predictions.
        """

        # add a column of ones to X to include the intercept in the model
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # make predictions
        y_pred = np.dot(X, self.optimized_weights)

        return y_pred
    

    def save_weights(self, path):
        """
        Save weights.
        """

        np.save(path, self.optimized_weights)

    
    def load_weights(self, path):
        """
        Load weights.
        """

        self.optimized_weights = np.load(path)