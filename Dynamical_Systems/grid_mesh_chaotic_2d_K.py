import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def func(k_val, t_val):
    t = smp.symbols("t")
    m = smp.symbols("m")
    k = smp.symbols("k")

    #b= 0.2

    #IMPPPPP these are coords, not displacements about mean, the PE breaks down or elseee
    x1, x2, x3, x4 = smp.symbols("x1 x2 x3 x4", cls = smp.Function)
    y1, y2, y3, y4 = smp.symbols("y1 y2 y3 y4", cls = smp.Function)

    L0 = 1 #length of the springs (they are all the same)
    L0_diag = L0*smp.sqrt(2)

    #Masses are like :  m1--m2
    #                   |   |
    #                   m3--m4

    x1 = x1(t)
    x2 = x2(t)
    x3 = x3(t)
    x4 = x4(t)

    y1 = y1(t)
    y2 = y2(t)
    y3 = y3(t)
    y4 = y4(t)

    x1_d = smp.diff(x1,t)
    x2_d = smp.diff(x2,t)
    x3_d = smp.diff(x3,t)
    x4_d = smp.diff(x4,t)

    y1_d = smp.diff(y1,t)
    y2_d = smp.diff(y2,t)
    y3_d = smp.diff(y3,t)
    y4_d = smp.diff(y4,t)

    x1_dd = smp.diff(x1_d,t)
    x2_dd = smp.diff(x2_d,t)
    x3_dd = smp.diff(x3_d,t)
    x4_dd = smp.diff(x4_d,t)

    y1_dd = smp.diff(y1_d,t)
    y2_dd = smp.diff(y2_d,t)
    y3_dd = smp.diff(y3_d,t)
    y4_dd = smp.diff(y4_d,t)


    #Massive discovery, PE is not simple, you need absolute coordinates and use pythagoras, dx dy wont do...


    def get_ke(m,vx,vy):
        temp = 0.5*m*(vx**2 + vy**2)
        return temp

    def get_pe(k,dx,dy):
        temp = 0.5*k*(dx**2+dy**2)
        return temp

    def get_disp(x1,x2,y1,y2,l):
        return smp.sqrt((x1-x2)**2+(y1-y2)**2) - l

    ke = sum([get_ke(*x) for x in [(m,x1_d,y1_d), (m,x2_d,y2_d), (m,x3_d,y3_d), (m,x4_d,y4_d)]])
    #pe = [get_pe(*x) for x in [(k1,(x1,y1,x2,y2)),(k2,x2-x1,y2-y4),(k3,x3-x4,y3-y1),(k4,x4-x3,y4-y2)]]


    pe1 = 0.5*k*(get_disp(x1,x2,y1,y2,L0)**2)
    pe2 = 0.5*k*(get_disp(x2,x4,y2,y4,L0)**2)
    pe3 = 0.5*k*(get_disp(x1,x3,y1,y3,L0)**2) 
    pe4 = 0.5*k*(get_disp(x3,x4,y3,y4,L0)**2)

    #TODO check if solids interact diagonally
    #extra pe terms for the diagonal, gives more rigidity, diagonals are also going to interact in this assumption
    #this assumption makes sense if it is a tight solid (maybe)
    pe5 = 0.5*k*(get_disp(x1,x4,y1,y4,L0_diag)**2)
    pe6 = 0.5*k*(get_disp(x2,x3,y2,y3,L0_diag)**2)

    pe = pe1+pe2+pe3+pe4+pe5+pe6

    Lag = ke - pe




    #this is gemini's recommended way to handle 8 ELs. no credit taken for this
    coords = [x1, y1, x2, y2, x3, y3, x4, y4]
    vels = [x1_d, y1_d, x2_d, y2_d, x3_d, y3_d, x4_d, y4_d]
    accels = [x1_dd, y1_dd, x2_dd, y2_dd, x3_dd, y3_dd, x4_dd, y4_dd]

    # Generate the 8 equations
    EL = [smp.diff(smp.diff(Lag, v), t) - smp.diff(Lag, c) for v, c in zip(vels, coords)]


    EL = [smp.simplify(mem) for mem in EL]
    sols = smp.solve(EL, accels)

    sols[x1_dd].args

    args = (x1,x2,x3,x4,
            y1,y2,y3,y4,
            x1_d,x2_d,x3_d,x4_d,
            y1_d,y2_d,y3_d,y4_d,
            m,
            k)

    funcs = [sols[x1_dd],sols[x2_dd],sols[x3_dd],sols[x4_dd],sols[y1_dd],sols[y2_dd],sols[y3_dd],sols[y4_dd]]

    sols_f = smp.lambdify(args, funcs, "numpy")

    #remember to flatten after solving !!!

    # def mag(a,b):
    #     return np.sqrt(a**2 + b**2)

    ###My state vector = (x1,x2,x3,x4,
                        # y1,y2,y3,y4,
                        # x1_d,x2_d,x3_d,x4_d,
                        # y1_d,y2_d,y3_d,y4_d,)
    def dsdt(state, t, m, k, b):
        acc_values_cols = sols_f(*state,m,k)

        acc_values_list = np.array(acc_values_cols).flatten()

        x1_d = state[8]
        x2_d = state[9]
        x3_d = state[10]
        x4_d = state[11]

        y1_d = state[12]
        y2_d = state[13]
        y3_d = state[14]
        y4_d = state[15]

        x1_dd = acc_values_list[0] - b*x1_d/m
        x2_dd = acc_values_list[1] - b*x2_d/m
        x3_dd = acc_values_list[2] - b*x3_d/m
        x4_dd = acc_values_list[3] - b*x4_d/m

        y1_dd = acc_values_list[4] - b*y1_d/m
        y2_dd = acc_values_list[5] - b*y2_d/m
        y3_dd = acc_values_list[6] - b*y3_d/m
        y4_dd = acc_values_list[7] - b*y4_d/m


        return x1_d, x2_d, x3_d, x4_d, y1_d, y2_d, y3_d, y4_d, x1_dd, x2_dd, x3_dd, x4_dd, y1_dd,y2_dd, y3_dd, y4_dd

    t = t_val
    k_v = k_val
    b_v = 0.2
    m_v = 1

    xcoords_list_init = (0,1,0,1)
    ycoords_list_init = (0,0,-1,-1)

    xvel_list_init = (0,0,0,0.5)
    yvel_list_init = (0,0,0,0.2) 

    initial = (*xcoords_list_init, *ycoords_list_init, *xvel_list_init, *yvel_list_init)

    solutions = odeint(dsdt, initial, t=t, args=(m_v, k_v, b_v) )


    x1_c = solutions[:,0]
    x2_c = solutions[:,1]
    x3_c = solutions[:,2]
    x4_c = solutions[:,3]

    y1_c = solutions[:,4]
    y2_c = solutions[:,5]
    y3_c = solutions[:,6]
    y4_c = solutions[:,7]

    # vx1_list = solutions[:,8]
    # vx2_list = solutions[:,9]
    # vx3_list = solutions[:,10]
    # vx4_list = solutions[:,11]

    # vy1_list = solutions[:,12]
    # vy2_list = solutions[:,13]
    # vy3_list = solutions[:,14]
    # vy4_list = solutions[:,15]

    vel_list = solutions[:,8:]

    KE = 0.5*np.sum(m_v*vel_list**2,axis = 1)

    PE1= 0.5*k_v*(np.sqrt(((x1_c - x2_c)**2 + (y1_c-y2_c)**2))-L0)**2
    PE2 = 0.5*k_v*(np.sqrt((x2_c-x4_c)**2 + (y2_c-y4_c)**2)-L0)**2
    PE3 = 0.5*k_v*(np.sqrt((x3_c-x4_c)**2 + (y3_c-y4_c)**2)-L0)**2
    PE4 = 0.5*k_v*(np.sqrt((x1_c-x3_c)**2 + (y1_c-y3_c)**2)-L0)**2

    PE5 = 0.5*k_v*(np.sqrt((x1_c-x4_c)**2 + (y1_c-y4_c)**2)-L0_diag)**2
    PE6 = 0.5*k_v*(np.sqrt((x2_c-x3_c)**2 + (y2_c-y3_c)**2)-L0_diag)**2

    PE = PE1+PE2+PE3+PE4+PE5+PE6

    TotalE = KE + PE

    return TotalE

    # plt.figure()
    # plt.loglog(t[1:],TotalE[1:], label = "Total energy decay of the system")
    # plt.title(f"Energy DEcay for Spring Const k = {k_v}")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    t_val = np.linspace(0,50,1000)
    k_vals = [0.01, 0.1, 1 , 10, 100]
    TE = []
    for k_val in k_vals:
        TE.append(func(k_val,t_val))

    plt.figure()
    for i,k_val in enumerate(k_vals):
        plt.loglog(t_val[1:],TE[i][1:],label = f"For K = {k_val}" )
        plt.legend()

    plt.show()
