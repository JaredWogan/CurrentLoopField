from scipy.integrate import quad
import numpy as np
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 1000
from matplotlib import pyplot as plt

k = 50
n = 2*k+1

u = 1
I = 1
R = 1

def Bx(p,x,y,z):
    return (z * np.cos(p)) / ( x**2 + y**2 + z**2 + R**2 - 2*R*(x*np.cos(p) + y*np.sin(p)) )**(3/2)


def By(p,x,y,z):
    return (z * np.sin(p)) / ( x**2 + y**2 + z**2 + R**2 - 2*R*(x*np.cos(p) + y*np.sin(p)) )**(3/2)


def Bz(p,x,y,z):
    return (R - y*np.sin(p) - x*np.cos(p)) / ( x**2 + y**2 + z**2 + R**2 - 2*R*(x*np.cos(p) + y*np.sin(p)) )**(3/2)


x = np.linspace(-2*R, 2*R, n)
y = np.linspace(-2*R, 2*R, n)
z = np.linspace(-2*R, 2*R, n)

B_x = [[[ 0 for i in range(n) ] for j in range(n) ] for k in range(n) ]
B_y = [[[ 0 for i in range(n) ] for j in range(n) ] for k in range(n) ]
B_z = [[[ 0 for i in range(n) ] for j in range(n) ] for k in range(n) ]


# Finding the index in the x-coordinate array corresponding to half the radius
# (in-case it isn't k should I change the domain)
kR2 = k
err = 100
for i in range(n):
    if abs(x[i] - R/2) < err :
        kR2 = i
        err = abs(x[kR2] - R/2)

# Finding the index in the x-coordinate array corresponding to 0 (incase it isn't k should I change the domain)
k0 = k
err = 100
for i in range(n):
    if abs(x[i]) < err:
        k0 = i
        err = abs(x[k0])

for i in range(n):
    print('i =', i)
    for j in range(n):
        for k in range(n):
            B_x[i][j][k] = quad( Bx,0,2*np.pi,args=(x[i],y[j],z[k]) )[0]
            B_y[i][j][k] = quad( By,0,2*np.pi,args=(x[i],y[j],z[k]) )[0]
            B_z[i][j][k] = quad( Bz,0,2*np.pi,args=(x[i],y[j],z[k]), limit=2000 )[0]


B_x_R2z = []
B_z_s0 = []

# Extracting the z=0 component, then taking the y=0 component
for y in B_z[:][:][k0]:
    B_z_s0.append(y[k0])

# Extracting the x=R/2 component, then taking the y=0 component
y = B_x[kR2][:][:]
B_x_R2z = y[k0]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.plot(x, B_z_s0)
ax1.set_title('B_z_s0')
ax2.plot(z, B_x_R2z)
ax2.set_title('B_x_R2z')

ax1.set(xlabel='x - coordinate', ylabel='B_z(s,0,0)', title='PHYS 3P36 - Assignment 2 - Question 2c)')
ax1.grid()

ax2.set(xlabel='z - coordinate', ylabel='B_x(R/2,0,z)', title='PHYS 3P36 - Assignment 2 - Question 2c)')
ax2.grid()

plt.show()

print('Done')



"""

Messing around with vector field plots

fig = plt.figure()
ax = plt.subplot()

print(y[kR2])
print(B_x[:][kR2][:])


ax.add_patch( plt.Circle((R,0), 0.05, color='blue') )
ax.add_patch( plt.Circle((-R,0), 0.05, color='blue') )


X, Z = np.meshgrid(np.array(x), np.array(z))

B_X = np.array(B_x[:][k][:])
B_Z = np.array(B_z[:][k][:])
for i in range(n):
    N = np.sqrt( B_X[i][:]**2 + B_Z[i][:]**2 )
    B_X[i][:] = B_X[i][:] / N
    B_Z[i][:] = B_Z[i][:] / N


print(B_X)

ax.quiver(X, Z, B_X, B_Z, units='xy')
plt.savefig("T.png", dpi=1000)

plt.show()
"""
