### This uses the same fundamental maths and concepts as the other but is sped up via multiprocessing

import multiprocessing
from multiprocessing import Pool
import queue
import pandas as pd
import math
import numpy as np

# Multiprocessing requires some function to run across multiple processes
# calcGradient is adjusted to take one argument due to Pool's requirments
def calcGradient(parameters):
    chunk, thetas = parameters
    n_features = len(thetas)

    gradients = np.zeros(n_features)

    for row in chunk:
        x = row[0] # extract features
        y = row[:1] # extract label

        # Performs dot product of every x value in the chunk and the thetas
        prediction = np.dot(x, thetas)

        # Subtract true value from predictions
        error = prediction - y

        gradients += error * x
    return gradients

# Run a loop till the maximum number of iterations is exceeded
def regression(data, alpha, MaxIter, threshold, thetas):
    thetas = np.array(thetas, dtype=float)
    n = len(data)

    num_cpus = multiprocessing.cpu_count()
    pool = Pool(num_cpus)

    # Split data into chunks for multiprocessing
    chunks = np.array_split(data, num_cpus)

    for i in range(MaxIter):
        # Convert inputs into format for pooling
        inputs = [(chunk, thetas) for chunk in chunks]

        # Now the actual multiprocessing happens
        results = pool.map(calcGradient, inputs)

        # Calculate the average of the gradients
        total_gradient = np.sum(results, axis = 0) / n

        # Update parameters
        new_thetas = thetas - alpha * total_gradient

        # Check for convergence by comparing step threshold
        if np.linalg.norm(new_thetas - thetas) < threshold:
            print(f"Convergence at iteration {i}")
            break

        thetas = new_thetas

    # Close and clean up the pool
    pool.close()
    pool.join()
    return thetas

def prep_data(path):
    df = pd.read_csv(path)

    # Force numeric everywhere
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Normalize features (exclude target)
    feature_cols = df.columns[1:]
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

    # Split explicitly
    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)  # features
    y = df.iloc[:, 0].to_numpy(dtype=np.float64)  # target

    return X, y

# Since we're performing multiprocessing we need a 'main' for it to find
if __name__ == '__main__':
    X, y = prep_data('mpg.csv')

    thetas = np.zeros(X.shape[1])  # automatically correct size

    result = regression(
        X,
        y,
        alpha=1e-6,
        max_iter=10000,
        threshold=1e-6,
        thetas=thetas
    )

    print("Final thetas:", result)