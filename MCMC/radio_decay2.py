import numpy as np
import matplotlib.pyplot as plt


time_limit = 100
N0 = 100000

p1 = 0.02   #A to B
p2 = 0.05   #B to C
p3 = 0.001  #B to A (reverse)
p4 = 0.008  #A directly to C

# lamb1 = -np.log(1-p1)
# lamb2 = -np.log(1-p2)

A = np.zeros(time_limit)
B = np.zeros(time_limit)
C = np.zeros(time_limit)

A[0] = N0
B[0] = 0
C[0] = 0

ba_count = 0
for i in range(1,time_limit):

    arand = np.random.uniform(0,1,int(A[i-1]))
    ab_truth = arand<= p1
    ac_truth = (arand>p1) & (arand <= p1+p4)

    ab_count = sum(ab_truth)
    ac_count = sum(ac_truth)

    brand = np.random.uniform(0,1,int(ab_count+B[i-1]))
    bc_truth = brand <= p2
    ba_truth = (brand>p2) & (brand < p2+p3)

    bc_count = sum(bc_truth)
    ba_count = sum(ba_truth)

    A[i] = A[i-1]-ab_count-ac_count+ba_count
    B[i] = B[i-1]-bc_count-ba_count+ab_count
    C[i] = C[i-1]+ac_count+bc_count

#     Transition matrix
#    0.972, 0.2, 0.008   
#    0.001, 0.949, 0.05
#     0,    0,     1
#

#!!!!!!!BE EXTREMELYCAREFUL WITH THE TRANSITION MATRIX FORM, AND COLUMN/ROW MAJOR STATE VECTOR. DOUBLE/TRIPLE CHECK.

trans_mat = np.array([[0.972, 0.02, 0.008],[0.001, 0.949, 0.05],[0, 0, 1]])


actual_states = []
initial = np.array([A[0],B[0],C[0]])
cur = initial

for i in range(time_limit):
    cur = cur @ trans_mat
    actual_states.append(cur)

actual_states= np.array(actual_states)

A_list = actual_states[:,0]
B_list = actual_states[:,1]
C_list = actual_states[:,2]


plt.figure()
plt.subplot(2,2,1)
plt.plot(A,label = "Montecarlo for A")
plt.plot(A_list, label = "Actual A values using TransMat", alpha = 0.7)
plt.legend()
plt.subplot(2,2,2)
plt.plot(B,label = "Monte Carlo for B")
plt.plot(B_list, label = "Actual B using TransMat", alpha = 0.7)
plt.legend()
plt.subplot(2,2,3)
plt.plot(C, label = "Monte Carlo for C")
plt.plot(C_list, label = "Actual C using TransMat", alpha = 0.7)

plt.legend()
plt.show()


