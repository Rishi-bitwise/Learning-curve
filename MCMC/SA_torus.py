import numpy as np

#Area function = (R+rcos(the)*r)

R = 1
r = 0.5
iters = 1000000

the = np.random.uniform(0,2*np.pi,iters)

tot = sum((R+r*np.cos(the))*r)
avg = tot/iters

domain_2d = 2*np.pi*2*np.pi

sa = avg*domain_2d
print(sa)