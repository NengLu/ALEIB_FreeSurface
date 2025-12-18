#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld as uw
import math
from underworld import function as fn
import numpy as np
import os

from underworld import UWGeodynamics as GEO
u = GEO.UnitRegistry
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size


# In[2]:


# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-3
GEO.rcParams['initial.nonlinear.max.iterations'] = 100
GEO.rcParams["nonlinear.tolerance"] = 1e-3
GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 16
GEO.rcParams["swarm.particles.per.cell.2D"] = 16
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1


# In[3]:


# scaling 3: vel
half_rate = 1.0 * u.centimeter / u.year
model_length = 500. * u.kilometer
gravity = 9.81 * u.meter / u.second**2
bodyforce = 3300 * u.kilogram / u.metre**3 *gravity 

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM


# In[4]:


xmin, xmax = ndim(-250 * u.kilometer), ndim(250 * u.kilometer)
ymin, ymax = ndim(-500 * u.kilometer), ndim(0 * u.kilometer)
yint = 0.

dy = ndim(500 * u.kilometer)/50
dx = dy
xRes,yRes = int((xmax-xmin)/dx),int((ymax-ymin)/dy)
yResa,yResb =int((ymax-yint)/dy),int((yint-ymin)/dy)

#tRatio =  int(sys.argv[1])
tRatio = 100
save_every = 1
use_fssa = False
if use_fssa:
    outputPath = "op_Ex_2DTopoRelax_FreeSurf_ALE_withFSSA0.5_yres{:n}_tRatio{:n}_Swarm_uwg".format(yRes,tRatio)
    save_every = 1
else:
    outputPath = "op_Ex_2DTopoRelax_FreeSurf_ALE_noFSSA_yres{:n}_tRatio{:n}_Swarm_uwg".format(yRes,tRatio)
    save_every = 5


# In[5]:


Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -gravity),
                  periodic=(True, False))
Model.outputDir= outputPath
Model.minStrainRate = 1e-18 / u.second


# In[6]:


wRatio = 1
D = np.abs(ymin)
Lambda = D/wRatio
k = 2.0 * np.pi / Lambda
mu0 = ndim(1e21  * u.pascal * u.second)
g = ndim(gravity)
rho0 = ndim(3300* u.kilogram / u.metre**3)
drho = rho0-0.
w_m = ndim(10*u.kilometer)

tau0 = 2*k*mu0/drho/g
tau = (D*k+np.sinh(D*k)*np.cosh(D*k))/(np.sinh(D*k)**2)*tau0

def perturbation(x):
    return w_m * np.cos(2.*np.pi*(x)/Lambda)

fn_coord = fn.input()
deform_fn = w_m * fn.math.cos(2.*np.pi*fn_coord[0]/Lambda)

# def get_analytical(x0,load_time):
#     #tau = (D*k+np.sinh(D*k)*np.cosh(D*k))/(np.sinh(D*k)**2)*tau0
#     #A = -F0/k/tau0
#     #B = -F0/k/tau
#     #C = F0/tau
#     #E = F0/tau/np.tanh(D*k)
#     #phi = np.sin(k*x)*np.exp(-tmax/tau)*(A*np.sinh(k*z)+B*np.cosh(k*z)+C*z*np.sinh(k*z)+E*z*np.cosh(k*z))
#     w = w_m*np.exp(-load_time/tau)
#     return w

max_time =  tau*4
dt_set = tau/tRatio


# In[7]:


# D_ib = yint
# axis_ib = np.where((Model.mesh.data[:,1]<=D_ib+dy/4)&(Model.mesh.data[:,1]>=D_ib-dy/4))
# Sets_ib = Model.mesh.specialSets["Empty"]
# for index in axis_ib:
#     Sets_ib.add(index)

minCoord = tuple([GEO.nd(val) for val in Model.minCoord])
maxCoord = tuple([GEO.nd(val) for val in Model.maxCoord])

init_mesh = uw.mesh.FeMesh_Cartesian(elementType=Model.elementType,
                                    elementRes=Model.elementRes,
                                    minCoord=minCoord,
                                    maxCoord=maxCoord,
                                    periodic=Model.periodic)

TField = init_mesh.add_variable(nodeDofCount=1)
TField.data[:, 0] = init_mesh.data[:, 1].copy()

top = Model.top_wall
bottom = Model.bottom_wall
conditions = uw.conditions.DirichletCondition(variable=TField,indexSetsPerDof=(top + bottom,))
system = uw.systems.SteadyStateHeat(
    temperatureField=TField,
    fn_diffusivity=1.0,
    conditions=conditions)
solver = uw.systems.Solver(system)

TField.data[top, 0] = perturbation(init_mesh.data[top,0])  

solver.solve()
with Model.mesh.deform_mesh():
     Model.mesh.data[:, -1] = TField.data[:, 0].copy()


# In[8]:


# if size == 1:
#     from underworld import visualisation as vis
#     Fig = vis.Figure(resolution=(500,500),rulers=True,margin = 20)
#     Fig.Mesh(Model.mesh)
#     Fig.show()
#     Fig.save('mesh0.png') 


# In[9]:


from underworld.swarm import Swarm
from collections import OrderedDict
Model.swarm_variables = OrderedDict()
Model.swarm = Swarm(mesh=Model.mesh, particleEscape=True)
Model.swarm.allow_parallel_nn = True
if Model.mesh.dim == 2:
    particlesPerCell = GEO.rcParams["swarm.particles.per.cell.2D"]
else:
    particlesPerCell = GEO.rcParams["swarm.particles.per.cell.3D"]
Model._swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(
    swarm=Model.swarm,
    particlesPerCell=particlesPerCell)

Model.swarm.populate_using_layout(layout=Model._swarmLayout)
Model._initialize()


# In[10]:


materialAShape = fn_coord[1] > deform_fn #GEO.shapes.Layer(top=Model.top, bottom=Model.bottom)
materialMShape = fn_coord[1] <= xmax+w_m*5

#materialA = Model.add_material(name="Air", shape=materialAShape)
materialM = Model.add_material(name="Mantle", shape=materialMShape)


# In[11]:


# if uw.mpi.rank == 0:
#     from underworld import visualisation as vis
#     fig_res = (500,500)

#     Fig = vis.Figure(resolution=fig_res,rulers=False,margin = 20,rulerticks=7,quality=2,clipmap=False) 
#     Fig.Points(Model.swarm, Model.materialField,fn_size=2.0,discrete=True,colourBar=True,colours='blue orange')
#     Fig.Mesh(Model.mesh)
#     Fig.show()
#     Fig.save("Modelsetup.png")


# In[12]:


# Model.maxViscosity = 1e21 * u.pascal * u.second
# Model.minViscosity = 1e18 * u.pascal * u.second

#materialA.viscosity = 1e18 * u.pascal * u.second
materialM.viscosity = 1e21 * u.pascal * u.second

#materialA.density = 0.
materialM.density = 3300 * u.kilogram / u.metre**3


# In[13]:


Model.set_velocityBCs(left=[0.,None],right=[0,None],bottom=[0.,0.], top=[None,None])

Model.init_model()

Model.solver.set_inner_method("mumps")
Model.solver.set_penalty(1e3)


# In[14]:


Model.freeSurface = True

# In[15]:


if use_fssa:
    Model._fssa_factor = 0.5


# In[16]:


max_time = dimen(tau*4,u.kiloyear)
dt_set = dimen(tau/tRatio,u.kiloyear)
checkpoint_interval = dt_set*save_every 

Model.run_for(max_time, checkpoint_interval=checkpoint_interval,dt=dt_set)
