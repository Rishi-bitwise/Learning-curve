#our worls  is tiny circles at poles (zcoords), and then big circle at the center,
#so obviously phi is 0 to pi, and theta is the rest of the circles that depends on phi
#rest of the circles trace out 0 to 2pi and radius of these is just Rsin(phi)


import numpy as np

#the_limit = 2*np.pi
phi_limit = np.pi
iters = 1000000
R = 1

#the_points = np.random.uniform(0,the_limit,iters)
phi_points = np.random.uniform(0,phi_limit,iters)

#our surface element is actually R*R*sin(phi)

se = R*R*np.sin(phi_points)

avg = np.average(se)
domain = 2*np.pi*np.pi

sa = domain*avg

print(sa)


