import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.loadtxt(sys.argv[1])

Nq = 3
size = data.shape[0]
Ne = int((size/Nq/Nq)**0.5)

x, y, rho = map(lambda v: data[:,v], (0,1,2))

#x, y, rho = map(lambda v: v.reshape(Nq,Nq,Ne,Ne).swapaxes(1,2).reshape(Nq*Ne,Nq*Ne), (x,y,rho))
#x, y, rho = map(lambda v: v.reshape(Ne,Nq,Ne,Nq).reshape(Nq*Ne,Nq*Ne), (x,y,rho))
#x, y, rho = map(lambda v: v.reshape(Nq*Ne,Nq*Ne), (x,y,rho))

#x, y, rho = map(lambda v: v.reshape(Nq*Ne,Nq*Ne), (x,y,rho))

#x, y, rho = map(lambda v: v.reshape(Nq,Nq,Ne,Ne), (x,y,rho))
#for e0 in range(Ne):
#    for e1 in range(Ne):
#        plt.contourf(x[:,:,e0,e1],y[:,:,e0,e1],rho[:,:,e0,e1])

x, y, rho = map(lambda v: v.reshape(Ne,Ne,Nq,Nq).swapaxes(1,2).reshape(Ne*Nq,Ne*Nq), (x,y,rho))
#x, y, rho = map(lambda v: v.reshape(Ne,Ne,Nq,Nq).swapaxes(0,2).swapaxes(1,2).swapaxes(2,3).reshape(Ne*Nq,Ne*Nq), (x,y,rho))

#Ne1,Ne2,Nq1,Nq2 -> Nq1,Ne2,Ne1,Nq2 -> Nq1,Ne1,Ne2,Nq2 -> Nq1,Ne1,Nq2,Ne2

plt.subplots(2,1)
plt.subplot(2,1,1)
plt.contourf(x,y,0.0023*(1+np.sin(x)))
plt.subplot(2,1,2)
plt.contourf(x,y,rho)
#for e0 in range(Ne):
#    for e1 in range(Ne):
#        plt.contourf(x[e0,e1,:,:],y[e0,e1,:,:],rho[e0,e1,:,:])

#plt.scatter(x[:64], y[:64])
plt.savefig('plot.pdf')

#plt.show()
