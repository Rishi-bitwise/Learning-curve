import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_covars(arr, lag_limit):

    avg = np.mean(arr)
    std = np.std(arr)
    covar_list = []
    for lag in tqdm(range(1,lag_limit)):
        cur_data_centered = arr[:-lag] - avg
        lag_data_centered = arr[lag:] - avg

        mult = cur_data_centered * lag_data_centered
        
        mult_avg = np.mean(mult)

        temp_covar = mult_avg/std**2
        covar_list.append(temp_covar)

    return covar_list



PATH = r"C:\Users\sapta\Desktop\practice\TIME_SERIES_DATA\pi.txt"
LAG_LIMIT_START = 100000
MIN_LIST_LEN = 10

with open(PATH,'r') as f:
    fileval = f.readline()


digits = np.array([int(ch) for ch in fileval])

print(len(digits))

avg = np.mean(digits)
std = np.std(digits)

#We now have a numpy array of the digits of PI, we now need autocorrelation with different lags.
#I have assumed the average/std is the same despite losing data due to lag...

covar = []
covar_list = np.zeros(LAG_LIMIT_START)

arr = digits
lag_lim = LAG_LIMIT_START

i = 0
while len(covar_list) > 1:
    covar_list = get_covars(arr, lag_lim)
    covar.append(covar_list)

    arr = covar_list
    lag_lim = lag_lim//10

    print(f"{i} iteration finished")
    i +=1



for i, clist in enumerate(covar):
    # 1. Create a NEW figure for every list in covar
    fig, ax = plt.subplots()
    
    # 2. Plot the data
    # We use range(len(clist)) to match the X and Y coordinates
    ax.plot(range(len(clist)), clist, 'o', markersize=2)
    
    # 3. Add details so you know which window is which
    ax.set_title(f"Autocorrelation Iteration: {i}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

# 4. Show ALL windows at the same time
plt.show()




        












    
