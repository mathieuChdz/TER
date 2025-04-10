import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import seaborn as sns
from matplotlib import pyplot as plt

import cupy as cp

import time

sns.set_theme()

## Generate data

def generate_data(n_samples=500, n_features=1, noise=30, bias=30, random_state=42):
    """
    Generate a regression dataset.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    noise : float
        Noise level.
    bias : float
        Bias term.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    X : ndarray
        Features of shape (n_samples, n_features).
    y : ndarray
        Target variable of shape (n_samples,).
    coeff : ndarray
        Coefficients of the linear model.
    """
    # Generate the data
    X, y, coeff = make_regression(n_samples=n_samples, n_features=n_features, 
                                  noise=noise, bias=bias, coef=True, 
                                  random_state=random_state)
    
    return X, y, coeff


def plot_train_train_sets(X_train, y_train, X_test, y_test):
    """
    Plot the train and test sets.
    """
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_train, y_train, marker = 'o', label="train data")
    plt.scatter(X_test, y_test, marker = 'o', label="test data")
    plt.xlabel("X", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Train and Test Sets", fontsize=16)
    
    plt.savefig("Graphs/LinearRegression_train_test_sets.jpg", bbox_inches='tight')
    plt.show()

# ## Model

class LinearRegression_CPU():
    """
    Implements the linear regression algorithm.
    Closed-form solution (not gradient descent)
    """
    
    def __init__(self):
        """
        Constructor.
        """
        
        self.beta_hat = None # solution de la regression
        
    
    def fit(self, X_train, y_train): # fit = entrainer
        """
        Fits the linear regression on the train set X_train, y_train.
        This method modifies the attribute beta_hat, which stores the solution of the LR. 
        
        Parameters
        ----------
        X_train : ndarray
            Train features of dim N x p, where p is the number of features.
        y_train : ndarray
            Train targets dim N x 1 (or N x q if multi-dimensional targets)
        """
        
        ones = np.ones(shape=(X_train.shape[0],1))
        X = np.concatenate([ones, X_train], axis=1)

        self.beta_hat = np.matmul(X.T, X)
        self.beta_hat = np.linalg.pinv(self.beta_hat)
        self.beta_hat = np.matmul(self.beta_hat, X.T)
        self.beta_hat = np.matmul(self.beta_hat, y_train)
        
    def predict(self, X):
        """
        Computes the predictions associated to features X.
        The predictions are computed as follows:
        - expand the features X with a first column of 1's
        - y_hat = X beta_hat
        
        Parameters
        ----------
        X : ndarray
            Features of dim N' x p, where p is the number of features.
            
        Returns
        -------
        y_hat : ndarray
            Predictions of dim N' x 1 (or N' x q if multi-dimensional targets)
        """
        
        ones = np.ones(shape=(X.shape[0],1))
        X = np.concatenate([ones, X], axis=1)

        return np.matmul(X, self.beta_hat)


class LinearRegression_GPU:
    """
    Implémente la régression linéaire en utilisant CuPy pour les calculs sur GPU.
    Solution en forme fermée (pas de descente de gradient).
    """
    
    def __init__(self):
        """
        Constructeur.
        """
        self.beta_hat = None  # Solution de la régression
    
    def fit(self, X_train, y_train):
        """
        Entraîne le modèle de régression linéaire sur le jeu de données d'entraînement.
        
        Parameters
        ----------
        X_train : ndarray (CuPy array)
            Caractéristiques d'entraînement de dimension N x p.
        y_train : ndarray (CuPy array)
            Cibles d'entraînement de dimension N x 1.
        """
        # Ajouter une colonne de 1 pour le biais
        ones = cp.ones(shape=(X_train.shape[0], 1))
        X = cp.concatenate([ones, X_train], axis=1)

        # Calcul de beta_hat en utilisant la formule fermée
        self.beta_hat = cp.matmul(X.T, X)
        self.beta_hat = cp.linalg.pinv(self.beta_hat)
        self.beta_hat = cp.matmul(self.beta_hat, X.T)
        self.beta_hat = cp.matmul(self.beta_hat, y_train)
    
    def predict(self, X):
        """
        Calcule les prédictions associées aux caractéristiques X.
        
        Parameters
        ----------
        X : ndarray (CuPy array)
            Caractéristiques de dimension N' x p.
            
        Returns
        -------
        y_hat : ndarray (CuPy array)
            Prédictions de dimension N' x 1.
        """
        # Ajouter une colonne de 1 pour le biais
        ones = cp.ones(shape=(X.shape[0], 1))
        X = cp.concatenate([ones, X], axis=1)

        # Calcul des prédictions
        return cp.matmul(X, self.beta_hat)

def create_lr(use_gpu=False):
    """
    Create a LinearRegression object.
    
    Parameters
    ----------
    use_gpu : bool
        If True, use GPU version of LinearRegression.
        
    Returns
    -------
    lr : LinearRegression or LinearRegression_GPU
        Linear regression object.
    """
    if use_gpu:
        return LinearRegression_GPU()
    else:
        return LinearRegression_CPU()

def plot_with_predictions(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test):
    """
    Plot the train and test sets along with the prediction line.
    
    Parameters
    ----------
    X_train : ndarray
        Training features.
    y_train : ndarray
        Training targets.
    X_test : ndarray
        Test features.
    y_test : ndarray
        Test targets.
    y_pred_train : ndarray
        Predicted values for the training set.
    y_pred_test : ndarray
        Predicted values for the test set.
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_train, y_train, marker='o', label="Train data", color="blue")
    plt.scatter(X_test, y_test, marker='o', label="Test data", color="orange")
    
    sorted_idx_train = np.argsort(X_train.flatten())
    sorted_idx_test = np.argsort(X_test.flatten())
    
    plt.plot(X_train[sorted_idx_train], y_pred_train[sorted_idx_train], label="Prediction (Train)", color="green")
    plt.plot(X_test[sorted_idx_test], y_pred_test[sorted_idx_test], label="Prediction (Test)", color="red")
    
    plt.xlabel("X", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Train and Test Sets with Prediction Line", fontsize=16)
    
    plt.savefig("Graphs/LinearRegression_with_predictions.jpg", bbox_inches='tight')
    plt.show()


def run_CPU(X_train, y_train, X_test, y_test):
    # -------------- CPU -------------- 
    lr_cpu = create_lr(use_gpu=False)

    # print("CPU --> beta_hat before fit: ", lr_cpu.beta_hat)

    # Fit the model
    lr_cpu.fit(X_train, y_train)
    # print("CPU --> beta_hat after fit: ", lr_cpu.beta_hat)

    # Predict
    y_hat = lr_cpu.predict(X_test)
    # print("CPU --> y_hat : ", y_hat)

    # plot_with_predictions(X_train, y_train, X_test, y_test, lr_cpu.predict(X_train), y_hat)

def run_GPU(X_train, y_train, X_test, y_test):
    # -------------- GPU -------------- 
    lr_gpu = create_lr(use_gpu=True)

    print("GPU --> beta_hat before fit: ", lr_gpu.beta_hat)
    # Fit the model
    lr_gpu.fit(cp.asarray(X_train), cp.asarray(y_train))
    print("GPU --> beta_hat after fit: ", lr_gpu.beta_hat)

    # Predict
    y_hat_gpu = lr_gpu.predict(cp.asarray(X_test))

    print("GPU --> y_hat : ", y_hat_gpu)
    y_hat_gpu = cp.asnumpy(y_hat_gpu)  # Convert back to NumPy array for plotting
    plot_with_predictions(X_train, y_train, X_test, y_test, cp.asnumpy(lr_gpu.predict(cp.asarray(X_train))), y_hat_gpu)

# Training and Results

if __name__ == "__main__":

    X, y, coeff = generate_data(n_samples=500, n_features=1, noise=30, bias=30, random_state=42)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.shape, y_train.shape, X_test.shape, y_test.shape

    plot_train_train_sets(X_train, y_train, X_test, y_test)

    # Fit and predict

    # run_CPU(X_train, y_train, X_test, y_test)

    # run_GPU(X_train, y_train, X_test, y_test)


    NB_ITERATION = 10
    
    result_global_mean_dict = {}
    for i in range(500000, 5000000, 500000):
        result_list_i = []
        # On fait N fois l'execution de l'algorithme pour le même nombre de points
        
        for _ in range(NB_ITERATION):
            print(".", end="")
            start = time.time()
            X, y, coeff = generate_data(n_samples=i, n_features=1, noise=30, bias=30, random_state=42)

            # train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train.shape, y_train.shape, X_test.shape, y_test.shape

            plot_train_train_sets(X_train, y_train, X_test, y_test)

            # Fit and predict

            run_CPU(X_train, y_train, X_test, y_test)
            end = time.time()
            result_list_i.append(end - start)


        result_global_mean_dict[i] = np.mean(result_list_i)
        print("mean time for ", i, " points: ", result_global_mean_dict[i])
    
    # On trace le temps d'execution en fonction du nombre de points
    plt.figure(figsize=(10, 6))
    plt.plot(list(result_global_mean_dict.keys()), list(result_global_mean_dict.values()), label="CPU")
    plt.xlabel("Number of points", fontsize=14)
    plt.ylabel(f"Execution time (s) (mean of {NB_ITERATION} iterations)", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Execution time of Linear Regression on CPU", fontsize=16)
    plt.savefig("Graphs/LinearRegression_CPU_execution_time.jpg", bbox_inches='tight')
        
















