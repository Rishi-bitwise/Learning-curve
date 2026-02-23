import sympy as smp
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#there is a spring mass system along X axis, and to that mass, we have a pendulum attached
#all physics revolves around the second derivative.

t,g,l1,k,l2,m1,m2 = smp.symbols("t g l1 k l2 m1 m2")

s,the = smp.symbols(r"s \theta", cls=smp.Function)
s = s(t)
the = the(t)

s_d = smp.diff(s,t)
the_d = smp.diff(the,t)

s_dd = smp.diff(s_d,t)
the_dd = smp.diff(the_d,t)

#symbols are done, we can now start with the Lagrangian

T1 = 0.5*m1*(s_d)**2

#BIG ERROR , I WROTE : T2 = 0.5*m2*(l2*the_d)**2 wrong....
#t2 also has to include motion due to the HO. Polar coordinates are tricky, the 
#point is also moving...


Vx = l2*the_d*smp.cos(the) + s_d
Vy = l2*the_d*smp.sin(the)
T2 = 0.5*m2*(Vx**2 + Vy**2)

T = T1 + T2

V = 0.5*k*s*s - m2*g*l2*smp.cos(the)

Lag = T - V

#We have the lagrangian, now get the ELs in symbolinc form, then solve it for acceleration

el1 = smp.diff(smp.diff(Lag,the_d),t) - smp.diff(Lag, the)
el2 = smp.diff(smp.diff(Lag,s_d),t) - smp.diff(Lag,s)

el1 = el1.simplify()
el2 = el2.simplify()

sols = smp.solve([el1,el2], (the_dd,s_dd))

thedd_f = smp.lambdify((t,g,l1,k,l2,m1,m2,the,s,the_d,s_d),sols[the_dd])
sdd_f = smp.lambdify((t,g,l1,k,l2,m1,m2,the,s,the_d,s_d),sols[s_dd])

def dsdt(pac,t,g,l1,k,l2,m1,m2):
    the_v,thed_v,s_v,sd_v = pac

    thedd_v = thedd_f(t,g,l1,k,l2,m1,m2,the_v,s_v,thed_v,sd_v)
    sdd_v = sdd_f(t,g,l1,k,l2,m1,m2,the_v,s_v,thed_v,sd_v)

    return thed_v, thedd_v, sd_v, sdd_v

t_v = np.linspace(0,40,1001)
g_v = 9.81
l1_v = 1
k_v = 5
l2_v = 1
m1_v = 2
m2_v = 2

initial = (1, 0, 0, 0)

solutions = odeint(dsdt,initial, t=t_v, args=(g_v,l1_v,k_v,l2_v,m1_v,m2_v))

the_coords = solutions[:,0]
s_coords = solutions[:,2]

ho_coords = l1_v+s_coords

px_coords = ho_coords+l2_v*np.sin(the_coords)
py_coords = -l2_v*np.cos(the_coords)


def update(frame):
    ho.set_data([0,ho_coords[frame]], [0,0])
    pend.set_data([ho_coords[frame],px_coords[frame]], [0,py_coords[frame]])

    bob1.set_data([ho_coords[frame]], [0])
    bob2.set_data([px_coords[frame]], [py_coords[frame]])

    return ho, pend, bob1, bob2

fig, ax = plt.subplots()

ax.set_xlim(-4,4)
ax.set_ylim(-4,2)

plt.grid()

ho, = ax.plot([0, ho_coords[0]], [0,0])
pend, = ax.plot([ho_coords[0],px_coords[0]],[0,py_coords[0]])

bob1, = ax.plot([ho_coords[0]], [0], 'o', markersize = 4*m1_v+5, color = "green")
bob2, = ax.plot([px_coords[0]], [py_coords[0]], 'o', markersize = 4*m2_v+5, color = "red")

mean_pos, = ax.plot([l1_v, l1_v], [0.5, 1], color = "red")

animation = FuncAnimation(fig,update, frames = len(t_v), interval = 50, blit= True )

plt.show()