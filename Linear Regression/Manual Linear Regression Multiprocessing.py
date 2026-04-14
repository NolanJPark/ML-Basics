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

# Since we're performing multiprocessing we need a 'main' for it to find
if __name__ == '__main__':
    # Basic variables and importing the dataset is moved into the main
    df = pd.read_csv('mpg.csv')
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df = df.dropna(subset=["horsepower"])
    feature_cols = df.columns[1:7]
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    df[feature_cols] = (df[feature_cols] - means) / stds
    data = df.to_numpy(dtype=float)
    alpha = 0.000001
    MaxIter = 10000
    threshold = 0.000001
    thetas = [15, 7, 2, 4, 9, 11, 6]
    regression(data, alpha, MaxIter, threshold, thetas)
    print("Final thetas:", thetas)