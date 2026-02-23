import numpy as np


#Volume =  (sqrt(x^2 + y^2) - R)^2  + z^2 <= r^2

R = 3
r = 1

def vol(x,y,z):
    '''args are numpy arrays, and return is numpy array.
    if return is negative then inside the torus, else outside/on the surface'''
    v = ((np.sqrt(x**2 + y**2)) - R)**2 + z**2 - r**2
    return v

xylimits = R+r
zlimits = r
iters = 1000000

x_points = np.random.uniform(-xylimits, xylimits, iters)
y_points = np.random.uniform(-xylimits, xylimits, iters)
z_points = np.random.uniform(-zlimits, zlimits, iters)

truth_mat = vol(x_points,y_points, z_points) <= 0

total_vol = 2*xylimits*2*xylimits*2*zlimits
inside_points = sum(truth_mat)
total_points = iters

vol_ratio = inside_points/total_points

torus_vol = vol_ratio*total_vol

print(torus_vol)

