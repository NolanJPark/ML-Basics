### This uses the same fundamental maths and concepts as the other but is sped up via multiprocessing

import multiprocessing
from multiprocessing import Pool
import queue
import pandas as pd
import math

# Multiprocessing requires some function to run across multiple processes
def calcGradient(theta: float, is_bias: int):
    # If we're calculating the gradient for the bias we need an edge case
    if is_bias == 0:


# Run a loop till the maximum number of iterations is exceeded
def regression(data, alpha, MaxIter, threshold, thetas):
    p = Pool(multiprocessing.cpu_count())
    processes = []

    # Begin gradient descent, this can't be a multiprocess since it needs to be done in order
    for i in range(MaxIter):
        # We first calculate the new gradients, and whis is where multiprocessing comes in
        input = ()
        p.map(calcGradient, )




# Since we're performing multiprocessing we need a 'main' for it to find
if __name__ == '__main__':
    # Basic variables and importing the dataset is moved into the main
    df = pd.read_csv('mpg.csv')
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df = df.dropna(subset=["horsepower"])
    print(df.dtypes)
    feature_cols = df.columns[1:7]
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    df[feature_cols] = (df[feature_cols] - means) / stds
    data = df.values
    print(data[0,])
    n = len(data)
    alpha = 0.000001
    MaxIter = 10000
    threshold = 0.000001
    thetas = [15, 7, 2, 4, 9, 11, 6]
    regression(data, alpha, MaxIter, threshold, thetas)