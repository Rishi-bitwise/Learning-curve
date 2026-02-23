import sympy as smp
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#Get the variables in symbolic notation first

t,g = smp.symbols("t g")
m1, m2 = smp.symbols("m_1 m_2")
L1, L2 = smp.symbols("L_1 L_2")

the1, the2 = smp.symbols(r"\theta_1 \theta_2", cls = smp.Function)

the1 = the1(t)
the2 = the2(t)

the1_d = smp.diff(the1,t)
the2_d = smp.diff(the2,t)

the1_dd = smp.diff(the1_d,t)
the2_dd = smp.diff(the2_d,t)

x1 = L1*smp.sin(the1)
x2 = x1 + L2*smp.sin(the2)

y1 = -L1*smp.cos(the1)
y2 = y1 - L2*smp.cos(the2)

#symbols are basically done. Now get the lagrangian

T1 = 0.5*m1*(smp.diff(x1,t)**2 + smp.diff(y1,t)**2)
T2 = 0.5*m2*(smp.diff(x2,t)**2 + smp.diff(y2,t)**2)

T = T1 + T2
V = m1*g*y1 + m2*g*y2

L = T - V

#lagrangian is done, now EL wrt the1 and the2, to get angular acceleration.
#we basically need to update the step using the1_d and the1_dd for Runge-Kutta

EL1 = smp.diff(smp.diff(L,the1_d),t) - smp.diff(L,the1)
EL2 = smp.diff(smp.diff(L,the2_d),t) - smp.diff(L,the2)

EL1 = EL1.simplify()
EL2 = EL2.simplify()

sols = smp.solve([EL1, EL2], (the1_dd,the2_dd))


#now we need the 'number' form of this, and not the symbolic stuff, so we to lambdify

el1_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the1_dd])
el2_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the2_dd])

#We now have the1_dd and the2_dd, we now need a function for odeint to step.

def dsdt(s,t,g,m1,m2,L1,L2):
    the1, the1_d, the2, the2_d = s

    t1dd_v = el1_f(t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d)
    t2dd_v = el2_f(t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d)

    return the1_d, t1dd_v, the2_d, t2dd_v


t_v = np.linspace(0,40,1001)
m1_v = 1
m2_v = 1
g_v = 9.81
L1_v = 1
L2_v = 1

initial = [1,2,2,2]

solutions_array = odeint(dsdt, initial, t=t_v, args=(g_v,m1_v,m2_v,L1_v,L2_v))

print(solutions_array.shape)

#Now extract the theta values, so we can get the actual coordinates
the1_vals = solutions_array[:,0]
the2_vals = solutions_array[:,2]

#calculate the coordinates using the angles
x1_coords = L1_v*np.sin(the1_vals)
y1_coords = -L1_v*np.cos(the1_vals)

x2_coords = x1_coords + L2_v*np.sin(the2_vals)
y2_coords = y1_coords - L2_v*np.cos(the2_vals)


#Now we start the animation function, this is the actual tricky part

def update(frame):
    pend1.set_data([0, x1_coords[frame]], [0, y1_coords[frame]])
    pend2.set_data([x1_coords[frame], x2_coords[frame]], [y1_coords[frame], y2_coords[frame]])
    
    mass1.set_data([x1_coords[frame]], [y1_coords[frame]])
    mass2.set_data([x2_coords[frame]], [y2_coords[frame]])

    return pend1, mass1, pend2, mass2



fig, ax = plt.subplots()
ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,1)
plt.grid()

print(type(x1_coords[0]))
print(type(y1_coords[0]))

pend1, = ax.plot([0, x1_coords[0]], [0, y1_coords[0]])
pend2, = ax.plot([x1_coords[0], x2_coords[0]], [y1_coords[0], y2_coords[0]])

mass1, = ax.plot([x1_coords[0]], [y1_coords[0]], 'o', markersize = 4*int(m1_v)+5, color = "green")
mass2, = ax.plot([x2_coords[0]], [y2_coords[0]], 'o', markersize = 4*int(m2_v)+5, color = "red")

animation = FuncAnimation(fig, update, frames = len(t_v), interval = 25, blit=True)
plt.show()
