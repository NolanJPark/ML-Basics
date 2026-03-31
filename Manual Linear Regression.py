import pandas as pd
import math
# dataset: mpg.csv
# learning rate: 0.001
# max iterations: 10,000
# minimum step threshold: 0.000001
df = pd.read_csv('mpg.csv')
alpha = 0.000001
MaxIter = 10000
threshold = 0.000001

# treat the dataset, turning a string to an int and normalize features
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower"])
print(df.dtypes)

feature_cols = df.columns[1:7]

# Save our means and std used for normalizing for future predictions
means = df[feature_cols].mean()
stds = df[feature_cols].std()

df[feature_cols] = (df[feature_cols] - means) / stds

# to speed up the program extract the raw data so the dataset doesn't have to be checked
data = df.values
print(data[0,])

# Hypothesis: y = b + w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
#   y: mpg
#   b: bias
#   w1: weight of cylinders
#   x1: cylinders
#   w2: weight of displacement
#   x2: displacement
#   w3: weight of horsepower
#   x3: horsepower
#   w4: weight of weight
#   x4: weight
#   w5: weight of acceleration
#   x5: acceleration
#   w6: weight of model year
#   x6: year
n = len(df)

# Random start values
thetas = [15, 7, 2, 4, 9, 11, 6]

# SSR loss function
def ssr(thetas):
    result = 0

    for i in range(n):
        yhat = thetas[0]
        for j in range(1, len(thetas)):
            yhat += thetas[j] * data[i][j+1]
        result += (data[i][0] - yhat) ** 2
    return result

# function that calculates the gradients for each of the thetas
def calcGradients(thetas):
    # create a list for the gradient of each theta
    gradients = [0]*len(thetas)

    # Handle the case of our bias, total being the sum of our residuals squared
    total = 0
    # Run through every observation and calculate our current theta's accuracy
    # The gradient for our bias is (2/n)sum(y_i - yhat_i) where n is the number of observations
    for i in range(n):
        # To calculate our yhat for i we first add the bias
        y_hat = thetas[0]

        # We then run through the other thetas (the weights for the features)
        for j in range(1, len(thetas)):
            # We add the theta times the input feature for i to our yhat
            y_hat += thetas[j] * data[i][j]
        # We then find the residual, or our true value minus our predicted
        total += (data[i][0] - y_hat)
    gradients[0] = total * (2/n)

    # The gradient for the weights of our features is (2/n)sum((y_i - yhat_i)xi)
    # We perform the same process as with our bias but when we calculate the sum we add in xi
    for i in range(1, len(thetas)):
        total = 0
        for j in range(n):
            y_hat = thetas[0]
            for k in range(1, len(thetas)):
                y_hat += thetas[k] * data[j][k]
            total += (data[j][0] - y_hat) * data[j][i]
        gradients[i] = total * (2/n)
    return gradients

# Run a loop till the maximum number of iterations is exceeded
for i in range(1, MaxIter + 1):

    # Get a list of our gradients
    gradients = calcGradients(thetas)
    # Calculate the steps we'll take for each theta by multiplying our gradients by our learning rate
    steps = [0]*len(gradients)
    for j in range(len(thetas)):
        steps[j] = gradients[j]*alpha

    # If all our steps breach the threshold we break out of the loop
    if all(abs(step) < threshold for step in steps):
        print(f"Convergence at iteration {i}")
        break

    # Calculate our new thetas by subtracting our step for that theta
    for j in range(len(thetas)):
        thetas[j] -= steps[j]

    # Print an update at some number of iterations
    if i % 1000000000000 == 0:
        print(f"Iteration {i}")
        for j in range(len(thetas)):
            print(f"Theta {j} = {thetas[j]}")
        print(f"SSR = {ssr(thetas)}")

    # print("Calculating...")

#print our final thetas as well as some assessment things like our MSE
print("\n\nFinal Parameters")
for j in range(len(thetas)):
    print(f"Theta {j} = {thetas[j]}")
print(f"SSR = {ssr(thetas)}")
print(f"True mpg: {data[0][0]}")
y = thetas[0]
for i in range(1, len(thetas)):
    y += thetas[i] * data[0][i]
print(f"predicted mpg: {y}")

def mse(thetas):
    total = 0
    for i in range(n):
        yhat = thetas[0]
        for j in range(1, len(thetas)):
            yhat += thetas[j] * data[i][j]

        total += (data[i][0] - yhat) ** 2

    return total / n

def rmse(thetas):
    return math.sqrt(mse(thetas))

def r_squared(thetas):
    # Mean of y
    mean_y = sum(data[i][0] for i in range(n)) / n

    ssr = 0
    sst = 0

    for i in range(n):
        yhat = thetas[0]
        for j in range(1, len(thetas)):
            yhat += thetas[j] * data[i][j]

        ssr += (data[i][0] - yhat) ** 2
        sst += (data[i][0] - mean_y) ** 2

    return 1 - (ssr / sst)

print("MSE:", mse(thetas))
print("RMSE:", rmse(thetas))
print("R^2:", r_squared(thetas))

# Calculate mpg of a 1994 supra
supra_data = pd.DataFrame({
    'cylinders': [6],
    'displacement': [3],
    'horsepower': [320],
    'weight': [3415],
    'acceleration': [4.6],
    'model year': [94],
})
supra_data[feature_cols] = (supra_data[feature_cols] - means) / stds
supra = (
    thetas[0]
    + thetas[1] * supra_data.loc[0, 'cylinders']
    + thetas[2] * supra_data.loc[0, 'displacement']
    + thetas[3] * supra_data.loc[0, 'horsepower']
    + thetas[4] * supra_data.loc[0, 'weight']
    + thetas[5] * supra_data.loc[0, 'acceleration']
    + thetas[6] * supra_data.loc[0, 'model year']
)
print(supra)