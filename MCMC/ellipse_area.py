import math as mt
import numpy as np
import matplotlib.pyplot as plt

#rotated only ellipse not displaced. 

a = 3
b = 1.5
the = np.deg2rad(30)
iter_limits = 1000000

#ellipse formula is : (xcost+ysint)^2/a^2 + (-xsint+ycost)^2/b^2 = 1

#we can se trig of (a,b) to get bounds. or just guess and take a large value
#obviously ellipse will not cross the double of a,b...
x_lim = 6
y_lim = 3


x_points = np.random.uniform(-x_lim,x_lim,iter_limits)
y_points = np.random.uniform(-y_lim,y_lim,iter_limits)

mat1 = ( (x_points*np.cos(the))  +  (y_points*np.sin(the)))**2/a**2
mat2 = ( (-x_points*np.sin(the))  +  (y_points*np.cos(the)))**2/b**2

final_mat = mat1+mat2

truth_mat = final_mat[final_mat<=1]  

inside_count = len(truth_mat)
total_count = len(final_mat)

area_ratio = inside_count/total_count

final_area = area_ratio*2*x_lim*2*y_lim

print(final_area)

