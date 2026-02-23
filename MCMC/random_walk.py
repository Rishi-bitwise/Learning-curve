import numpy as np
import matplotlib.pyplot as plt

total_steps = 10000

total_particles = 20000

positions = np.zeros((total_particles, total_steps))

for particle in range(total_particles):
    
    rand = np.random.choice([1,-1], size = total_steps)
    positions[particle] = np.cumsum(rand)

#average_drift = np.sum(positions, axis = 0)/total_particles       #this just averages to zero

average_drift = np.sqrt(np.mean(positions**2,axis = 0))

actual_drift = np.sqrt(np.arange(total_steps))

plt.figure()
plt.subplot(2,1,1)
plt.hist(positions[:,-1],bins = 50)
plt.legend()
plt.subplot(2,1,2)
plt.loglog(average_drift, label = "Simulated average drift over time")
plt.loglog(actual_drift, label = "Actual drift(is constant times sqrt(steps))")
plt.legend()
plt.show()

