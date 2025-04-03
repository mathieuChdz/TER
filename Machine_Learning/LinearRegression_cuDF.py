import cudf
from cuml.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def generate_data_cudf(n_samples=500, n_features=1, noise=30, bias=30, random_state=42):
    """
    Generate a regression dataset and return it as cuDF DataFrames.
    """
    
    X, y, coeff = make_regression(n_samples=n_samples, n_features=n_features, 
                                  noise=noise, bias=bias, coef=True, 
                                  random_state=random_state)
    # Convertir en cuDF DataFrames
    X_cudf = cudf.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y_cudf = cudf.Series(y)
    return X_cudf, y_cudf, coeff

def exec_linear_regression_cudf(X, y):
    """
    Execute linear regression using cuDF and cuML.
    """
    
    # On génère les données
    X, y, coeff = generate_data_cudf(n_samples=500, n_features=1, noise=30, bias=30, random_state=42)

    # Split du train set et du test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle
    lr_gpu = LinearRegression()

    # Fit du modèle
    lr_gpu.fit(X_train, y_train)

    # Prédiction sur le train set et le test set
    y_pred_train = lr_gpu.predict(X_train)
    y_pred_test = lr_gpu.predict(X_test)

    # On convertit les données en numpy pour la visualisation
    X_train_np = X_train.to_pandas().values.flatten()
    y_train_np = y_train.to_pandas().values
    X_test_np = X_test.to_pandas().values.flatten()
    y_test_np = y_test.to_pandas().values
    y_pred_train_np = y_pred_train.to_pandas().values
    y_pred_test_np = y_pred_test.to_pandas().values

    # plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_np, y_train_np, color="blue", label="Train data")
    plt.scatter(X_test_np, y_test_np, color="orange", label="Test data")
    plt.plot(np.sort(X_train_np), y_pred_train_np[np.argsort(X_train_np)], color="green", label="Prediction (Train)")
    plt.plot(np.sort(X_test_np), y_pred_test_np[np.argsort(X_test_np)], color="red", label="Prediction (Test)")
    plt.xlabel("X", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Linear Regression with cuDF and cuML", fontsize=16)
    plt.savefig("Graphs/LinearRegression_with_cuDF_cuML.jpg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Generate data and execute linear regression
    X, y = generate_data_cudf()
    exec_linear_regression_cudf(X, y)
    print("Linear regression executed successfully with cuDF and cuML.")