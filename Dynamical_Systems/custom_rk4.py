from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np


def rk4_single_step(func, t, y, step_size, args):
    k1 = func(t,y,args)
    k2 = func(t+step_size/2, y+k1*step_size/2,args)
    k3 = func(t+step_size/2, y+k2*step_size/2, args)
    k4 = func(t+step_size, y+k3*step_size,args)

    wavg = (k1+2*k2+2*k3+k4)/6

    return y+step_size*wavg


def solver(func, t0, y0, t_end, step_size, args):
    t_vals = [t0]
    y_vals = [y0]

    temp_t = t0
    temp_y = y0

    while temp_t < t_end:

        if temp_t+step_size > t_end:
            step_size = t_end-temp_t

        temp_y = rk4_single_step(func, temp_t, temp_y, step_size, args)
        temp_t += step_size

        y_vals.append(temp_y)
        t_vals.append(temp_t)

    return t_vals, y_vals
    

def funca(t, y, args):
    return -9.81

def funcv(t,y, args):
    return args[min(int(t), len(args)-1)]

def funcx(t,y,args):
    return args[0]


t_start = 0
t_end = 10
sx = 0
sy = 0
vx = 5
vy = 40
step_size = 1
args = []

vy_list = solver(funca, t_start, vy, t_end, step_size, args)

sy_list = solver(funcv, t_start, sy, t_end, step_size, vy_list[1])

sx_list = solver(funcx, t_start, sx, t_end, step_size, [vx])



def dsdt1(state, t):
    x, v = state
    return v, -9.81

def dsdt2(state, t, vel):
    x = state
    return vel


t_v = np.linspace(0,t_end,int(t_end/step_size)+1)

solsy = odeint(dsdt1, (sy,vy), t_v )
solsx = odeint(dsdt2, sx, t_v, args = (vx,))

fancy_y = solsy[:,0]
fancy_x = solsx[:]

plt.figure()
plt.plot(fancy_x, fancy_y, label = "Odeint solver")
plt.plot(sx_list[1], sy_list[1], label = "Custom RK4")
plt.title("COmpare my RK4 to odeint")
plt.legend()
plt.show()


