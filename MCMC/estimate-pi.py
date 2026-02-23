import random as rand
import matplotlib.pyplot as plt
from math import pi

PIcount = 1
nCount = 1
iters = 100001

pi_range = 3.1415*10/(3.1415+1)
print(pi_range)
print(10-pi_range)
print(pi_range/(10-pi_range))

pi_estimate = []

for i in range(iters):
    rnum = rand.uniform(0,10)
    if rnum>=0 and rnum<=pi_range:
        PIcount +=1
    else:
        nCount +=1
    if i%20 == 0:
        pi_estimate.append(PIcount/nCount)


plt.figure()
plt.plot(pi_estimate, label = "in range of 10")
plt.legend()

plt.show()


