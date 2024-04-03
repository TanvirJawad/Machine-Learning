import pandas as pd
import numpy as np

df = pd.read_csv('geyser.csv')
all_data = df.to_numpy()

def compute_cost(data, weights):
    pred = weights[0] + weights[1]*data[:,0]
    cost = 0.5 * np.sum(np.square(pred - data[:,1]))
    return cost

def compute_grad(data, weights):
    # Extract the features (durations) and target values (intervals) from the data
    durations = data[:, 0]  # Duration values
    intervals = data[:, 1]  # Interval values
    
    # Predictions based on the current weights
    predictions = weights[0] + weights[1] * durations
    
    # Errors between predictions and actual values
    errors = predictions - intervals
    
    # Partial derivatives
    dJ_dw0 = np.sum(errors)  # Derivative with respect to w0
    dJ_dw1 = np.sum(errors * durations)  # Derivative with respect to w1
    
    # Return a numpy array containing the partial derivatives
    return np.array([dJ_dw0, dJ_dw1])

def print_info(i, weights, cost):
    print("Iteration %i"%i)
    print("Weights are currently <%f, %f>"%(w[0], w[1]))
    print("Current cost: %f"%cost)

lr = 0.00001 #learning rate--keep this small
stop = False #becomes true when the cost stops decreasing
i = 1 #count how many iterations in we are
w = np.array([0.0, 0.0]) #initialize to zero
cost_decrease = np.inf
cost = np.inf

while(cost_decrease > 0.001):
    deriv = compute_grad(all_data, w)
    w -= lr*deriv
    newcost = compute_cost(all_data, w)
    cost_decrease = cost - newcost
    cost = newcost
    print_info(i, w, cost)
    i += 1
