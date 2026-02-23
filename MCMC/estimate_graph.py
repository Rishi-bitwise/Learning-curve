import numpy as np
import matplotlib.pyplot as plt

minval = 0
maxval = 10
iters = 1000000
bins = 20

def f(x):
    y = 0.3*x**3 - 5*x**2 + 24*x + 12
    return y


def approx_func1(iters):

    currv = (minval+maxval)/2
    x_custom = []

    for _ in range(iters):
        candidate = currv + np.random.uniform(-0.51, 0.51)

        if minval <= candidate <= maxval :

            if f(currv) < f(candidate):
                currv = candidate 

            else :
                prob = f(candidate)/f(currv)
                if np.random.uniform(0,1) <prob:
                    currv = candidate

        x_custom.append(currv)

    return x_custom


lpoints = np.linspace(minval, maxval, iters)

actual_y = f(lpoints)
area = np.trapezoid(actual_y, lpoints)
x_custom = approx_func1(iters)


plt.figure()
plt.plot(lpoints,actual_y/area, label = "Actual Function")
plt.hist(x_custom, bins = 20, density = True, alpha = 0.5)
plt.legend()
plt.show()
