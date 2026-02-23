#What is this ??

#Spring pendulum is basically-- instead of a string or a rod that the bob is suspended
#from, we replace it with a spring. Not to be confused with (vertical_spring + pendulum)

import sympy as smp
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t, g, m, l, k = smp.symbols("t g m l k")

s , the = smp.symbols(r"s \theta", cls= smp.Function)  #s is the change in spring length, theta is angle

s = s(t)
the = the(t)

the_d = smp.diff(the,t)
s_d = smp.diff(s,t)

the_dd = smp.diff(the_d,t)
s_dd = smp.diff(s_d,t)

#we now have the symbols, we need the KE and the PE. 
#KE can be in terms of polar coordinates or cartesian. Lets try polar(harder).

change_in_r_length = smp.diff(s,t)      #this is in r>>
change_in_r_direction = smp.diff(the,t)*(l+s)       #this is in theta>>

T = 0.5*m*(change_in_r_direction**2 + change_in_r_length**2)

V = 0.5*k*s**2 - m*g*(smp.cos(the)*(l+s))

Lag = T - V

#We now have the lagrangian for the system, so we need to get the ELs wrt s and the

el1 = smp.diff(smp.diff(Lag,the_d),t) - smp.diff(Lag,the)
el2 = smp.diff(smp.diff(Lag,s_d),t) - smp.diff(Lag,s)

el1 = el1.simplify()
el2 = el2.simplify()
#ELs are here, now we equate to zero and get the derivatives so we can step using odeint

sols = smp.solve([el1,el2], (the_dd, s_dd))

thedd_f = smp.lambdify((t,g,m,l,k,the,s,the_d,s_d), sols[the_dd])
sdd_f = smp.lambdify((t,g,m,l,k,the,s,the_d,s_d), sols[s_dd])

#we have the labdified exp,so passing args, we get the vals.now odeint time

def dsdt(pac,t,g,m,l,k):

    the_v,s_v,thed_v,sd_v = pac
    thedd_v = thedd_f(t,g,m,l,k,the_v,s_v,thed_v,sd_v)
    sdd_v = sdd_f(t,g,m,l,k,the_v,s_v,thed_v,sd_v)

    return thed_v, sd_v, thedd_v, sdd_v

t_v = np.linspace(0,40,1001)
g_v = 9.81
m_v = 1
l_v = 1
k_v = 10
initial = (1,1,2,2)

solutions = odeint(dsdt, initial, t=t_v, args=(g_v,m_v,l_v,k_v))

print(solutions.shape)

the_coords = solutions[:,0]
s_coords = solutions[:,1]

#For animation we need to get the x and y coords, not the polar ones

x_coords = np.sin(the_coords)*(l_v+s_coords)
y_coords = -np.cos(the_coords)*(l_v+s_coords)

def update(frame):
    spring1.set_data([0,x_coords[frame]], [0,y_coords[frame]])
    bob.set_data([x_coords[frame]],[y_coords[frame]])

    return spring1, bob

fig, ax = plt.subplots()
ax.set_xlim(-3,3)
ax.set_ylim(-3,1)

plt.grid()

spring1, = ax.plot([0,x_coords[0]], [0,y_coords[0]])
bob, = ax.plot([x_coords[0]],[y_coords[0]], 'o', markersize= 4*m_v+5, color="red")

animation = FuncAnimation(fig, update, frames = len(t_v), interval=25, blit = True)
plt.show()




