import numpy as np
import matplotlib.pyplot as plt

N0 = 100000
p = 0.05

time_limit = 100
decayed = 0
sim_N = np.zeros(101)
sim_N[0] = N0

for i in range(1,time_limit):
    decayed = 0
    for j in range(int(sim_N[i-1])):
        rand = np.random.uniform(0,1)

        if rand <= p:
            decayed += 1

    sim_N[i] = sim_N[i-1]-decayed

def decay_func(t):

    lamb = -np.log(1-p)

    ret = N0*np.e**(-lamb*t)
    return ret

tspace = np.linspace(0,time_limit,time_limit+1)

real_N = decay_func(tspace)

print(real_N)
print(tspace)
print(sim_N)

plt.figure()
plt.plot(tspace, real_N, label = "Formula based decay")
plt.plot(tspace, sim_N, label = "MC estimated decay")

plt.legend()
plt.show()
    
    


    