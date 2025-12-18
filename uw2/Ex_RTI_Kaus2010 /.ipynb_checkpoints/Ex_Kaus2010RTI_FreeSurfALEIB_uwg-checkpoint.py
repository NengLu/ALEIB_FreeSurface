#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld as uw
import math
from underworld import function as fn
import numpy as np
import os

from underworld import UWGeodynamics as GEO

from _freesurface_ALEIB import FreeSurfaceProcessor_ALEIB

u = GEO.UnitRegistry
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size

half_rate = 1.0 * u.centimeter / u.year
model_length = 500 * u.km
bodyforce = 3200 * u.kg / u.m**3 * 9.81 * u.m / u.s**2

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM


# In[2]:


gravity_vector = (0.0, -9.81 * u.m / u.s**2)
xRes,yRes = 50, 64
Model = GEO.Model(elementRes=(50, 64), 
                  minCoord=(-250. * u.km, -500. * u.km), 
                  maxCoord=(250. * u.km, 140. * u.km), 
                  gravity=gravity_vector)

use_fssa = False
if use_fssa:
    outputPath = "op_Ex_Kaus2010RTI_FreeSurfALEIB_withFSSA0.5_yres{:n}_uwg".format(yRes)
else:
    outputPath = "op_Ex_Kaus2010RTI_FreeSurfALEIB_noFSSA_yres{:n}_uwg".format(yRes)
Model.outputDir=outputPath

amplitude = GEO.non_dimensionalise(5*u.km)
period = GEO.non_dimensionalise(Model.length) #* 2.0
interface_depth = GEO.non_dimensionalise(100*u.km)
func = -interface_depth + (amplitude * fn.math.cos(2.0*np.pi*fn.input()[0]/period))

shape = fn.input()[1] < -interface_depth + (amplitude * fn.math.cos(2.0*np.pi*fn.input()[0]/period))

Air_material = Model.add_material(name="Air_material", shape=GEO.shapes.Layer(top=Model.top, bottom=0.*u.km))
HD_material = Model.add_material(name="HD_material", shape=GEO.shapes.Layer(top=0.*u.km, bottom=Model.bottom))
LD_material = Model.add_material(name="LD_material", shape=shape)

npoints = 250
coords = np.ndarray((npoints, 2))
coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
coords[:, 1] = -interface_depth + (amplitude * np.cos(2.0*np.pi*coords[:, 0]/period))
#surf_tracers = Model.add_passive_tracers(name="Surface",vertices=coords)
Model.add_passive_tracers(name="Interface", vertices=coords)

coords = np.ndarray((npoints, 2))
coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
coords[:, 1] = 0.0
#surf_tracers = Model.add_passive_tracers(name="Surface",vertices=coords)
Model.add_passive_tracers(name="Surface", vertices=coords)


Air_material.density = 0. * u.kilogram / u.meter**3
LD_material.density = 3200 * u.kilogram / u.meter**3
HD_material.density = 3300 * u.kilogram / u.meter**3
Air_material.viscosity = 1e18 * u.pascal * u.second
LD_material.viscosity = 1e20 * u.pascal * u.second
HD_material.viscosity = 1e21 * u.pascal * u.second

Model.set_kinematicBCs(left=[0., None],
                       right=[0., None],
                       top=[None, 0.],
                       bottom=[0.,0.])
#                       order_wall_conditions=[ "top", "left", "right", "bottom"])
#Model.freeSurface = True

D_ib = 0.0
dy = ndim(500 * u.kilometer)/50
axis_ib = np.where((Model.mesh.data[:,1]<=D_ib+dy/4)&(Model.mesh.data[:,1]>=D_ib-dy/4))
Sets_ib = Model.mesh.specialSets["Empty"]
for index in axis_ib:
    Sets_ib.add(index)

Model._freeSurface = True
Model.inter_wall = Sets_ib
Model._freeSurface = FreeSurfaceProcessor_ALEIB(Model)

if use_fssa:
    Model._fssa_factor = 0.5
Model.init_model(pressure="lithostatic")

Model.solver.set_inner_method("mumps")
#Model.solver.set_penalty(1e6)
#GEO.rcParams["initial.nonlinear.tolerance"] = 1e-6
# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-4
GEO.rcParams['initial.nonlinear.max.iterations'] = 200
GEO.rcParams["nonlinear.tolerance"] = 1e-4
GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 20
GEO.rcParams["swarm.particles.per.cell.2D"] = 20
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1

# outputss=['pressureField',
#          'strainRateField',
#          'velocityField',
#           'projTimeField',
#            'projMaterialField',
#          'projViscosityField',
#          'projStressField',
#          'projDensityField',
#          'projStressTensor',]
# GEO.rcParams['default.outputs']=outputss


# In[3]:


# from underworld import visualisation as vis

# Fig = vis.Figure(figsize=(500,500), title="Material Field", quality=2)
# Fig.Points(Model.swarm, Model.materialField, fn_size=2.0)
# Fig.Mesh(Model.mesh)
# Fig.Points(Model.Interface_tracers, pointSize=8.0)
# Fig.show()


# ## Solver options

# In[ ]:


if use_fssa:
    Model.run_for(6.0 * u.megayears, checkpoint_interval=100000*u.year,dt=50000*u.year)
else:
    Model.run_for(6.0 * u.megayears, checkpoint_interval=100000*u.year,dt=2500*u.year)


# In[ ]:


#Model.run_for(nstep=55, checkpoint_interval=1, dt=100000*u.year)
#Model.run_for(nstep=1100, checkpoint_interval=1, dt=5000*u.year)


# In[ ]:


# Fig = vis.Figure(figsize=(500,500), title="Viscosity Field (Pa.s)", quality=3)
# Fig.Mesh(Model.mesh)
# Fig.Points(Model.swarm, 
#            GEO.dimensionalise(Model.viscosityField, u.pascal * u.second),
#            logScale=True,
#            fn_size=3.0)
# Fig.VectorArrows(Model.mesh, Model.velocityField)
# Fig.show()


# In[ ]:




