import numpy as np

import sys
file = sys.argv[1]
data = np.loadtxt(file)
dim = int(sys.argv[2])

with open(file) as f:
    time = f.readline()
    vars = f.readline()

time = time.rstrip().replace("#","")
vars = vars.rstrip().replace("#","").split(",")
vars = vars[dim:]
vars = (['X', 'Y', 'Z'])[:dim] + vars
vars = [v.strip() for v in vars]
vars = [v.replace(":", "_") for v in vars]
Nq = 8

Np = int(np.ceil(data.shape[0]**(1./dim)))
Ne = int(Np/Nq)
NqT = Nq**3

def form(dat):
    if dim==2:
      return dat.reshape(Ne,Ne,Nq,Nq).swapaxes(1,2).reshape(Ne*Nq,Ne*Nq)
    if dim==3:
      return dat.reshape(Ne,Ne,Ne,Nq,Nq,Nq).swapaxes(1,3).swapaxes(2,3).swapaxes(3,4).reshape(Ne*Nq,Ne*Nq,Ne*Nq) #.swapaxes(0,2) #.swapaxes()
      #return dat.reshape(Ne**3,Nq**3).swapaxes(0,1).reshape(Nq,Nq,Nq,Ne,Ne,Ne).swapaxes(1,3).swapaxes(2,3).swapaxes(3,4).reshape(Nq*Ne,Nq*Ne,Nq*Ne)
      #return dat.reshape(Nq,Nq,Nq,Ne,Ne,Ne).swapaxes(1,3).swapaxes(2,3).swapaxes(3,4).reshape(Ne*Nq,Ne*Nq,Ne*Nq)


#(Ne,Ne,Ne,Nq,Nq,Nq) -> (Ne,Nq,Ne,Ne,Nq,Nq) -> (Ne,Nq,Ne,Ne,Nq,Nq)

import tecplot as tp
#tp.session.start_roaming(24)
from tecplot.constant import PlotType, Color
ds = tp.active_frame().create_dataset('Data at time: '+time, vars)
zone = ds.add_ordered_zone('Data', (Np, Np, Np)[:dim])
for i, var in enumerate(vars):
    zone.values(var)[:] = form(data[:,i]).ravel()

"""
#print(Np)
#Np = 2

vars = ['X', 'Y', 'Z']
ds = tp.active_frame().create_dataset('Data at time: '+time, vars)
N = 2
zone = ds.add_ordered_zone('Data', (Np, Np, Np)[:dim])

#def form(dat):
#    return dat.reshape(Ne**3,Nq**3).swapaxes(0,1).reshape(Ne,Ne,Ne,Nq,Nq,Nq).swapaxes(1,3).swapaxes(2,3).swapaxes(3,4).reshape(Ne*Nq,Ne*Nq,Ne*Nq) 

#def form(dat):
#  return dat.reshape(Ne**3,Nq**3)[:N,:].ravel()
  #return dat.reshape(Ne**3,Nq**3).swapaxes(0,1).reshape(Ne,Ne,Ne,Nq,Nq,Nq).swapaxes(1,3).swapaxes(2,3).swapaxes(3,4).reshape(Ne*Nq,Ne*Nq,Ne*Nq) 
  #return dat.reshape(Ne**3,Nq**3).T.ravel().reshape(Ne**3,Nq**3)

mp = [0,1,2]
#for i, var in enumerate(vars):
#  zone.values(var)[:] = form(data[:,mp[i]]).ravel()

x, y, z = map(form, (data[:,0], data[:,1], data[:,2]))

x_ = np.array([-1,0,0,1])
x, y, z = reversed(list(map(lambda v: v.ravel(), np.meshgrid(*[x_]*3, indexing='ij'))))
#print(Np, x.shape)

zone.values('X')[:] = x.ravel()
zone.values('Y')[:] = y.ravel()
zone.values('Z')[:] = z.ravel()

dat = [
 [-0.001, -0.001, -0.001],
 [ 0.   , -0.001, -0.001],
 [-0.001,  0.   , -0.001],
 [ 0.   ,  0.   , -0.001],
 [-0.001, -0.001,  0.   ],
 [ 0.   , -0.001,  0.   ],
 [-0.001,  0.   ,  0.   ],
 [ 0.   ,  0.   ,  0.   ],
 [ 0.   , -0.001, -0.001],
 [ 0.001, -0.001, -0.001],
 [ 0.   ,  0.   , -0.001],
 [ 0.001,  0.   , -0.001],
 [ 0.   , -0.001,  0.   ],
 [ 0.001, -0.001,  0.   ],
 [ 0.   ,  0.   ,  0.   ],
 [ 0.001,  0.   ,  0.   ]];

dat = np.array(dat)
zone.values('X')[:] = dat[:,0].ravel()
zone.values('Y')[:] = dat[:,1].ravel()
zone.values('Z')[:] = dat[:,2].ravel()
"""

#print(np.vstack([x,y,z]).T)
#exit(0)

#print(form(data[:,0]).ravel())

if dim==2:
  tp.active_frame().plot_type = tp.constant.PlotType.Cartesian2D
if dim==3:
  tp.active_frame().plot_type = tp.constant.PlotType.Cartesian3D

frame = tp.active_frame()
plot = frame.plot()
plot.activate()
plot.show_contour = True
#contour = tp.active_frame().plot().contour(0)
#print(dir(contour))
#contour.show = True

#tp.export.save_ps('plot.ps')
#import subprocess
#subprocess.call(['ps2pdf', 'plot.ps'])

from pyevtk.hl import gridToVTK
pointData = {var: form(data[:,i]) for i, var in enumerate(vars[dim:],start=dim)}
x, y, z = form(data[:,0]), form(data[:,1]), form(data[:,2])
if dim==2:
  #func = lambda v: np.stack(form(v).reshape(Nq*Ne,Nq*Ne), 2)
  func = lambda v: form(v).reshape(Nq*Ne,Nq*Ne,1)
  pointData = {var: func(data[:,i]) for i, var in enumerate(vars[dim:],start=dim)}
  x, y, z = [func(data[:,i]) for i in range(3)]
  z[:] = 1
gridToVTK(file, x, y, z, pointData=pointData)
tp.save_layout(file+".lpk", include_data=True)
#tp.save_layout(file+".lay")
