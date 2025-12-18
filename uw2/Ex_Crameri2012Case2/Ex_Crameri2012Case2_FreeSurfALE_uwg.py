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

half_rate = 1.0 * u.centimeter / u.year
model_length = 1000 * u.km
gravity = 10. * u.meter / u.second**2 
ref_density   = 3300 * u.kilogram / u.meter**3
bodyforce = ref_density * gravity

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM


# In[2]:


# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-4
GEO.rcParams['initial.nonlinear.max.iterations'] = 200
GEO.rcParams["nonlinear.tolerance"] = 1e-4
GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 36
GEO.rcParams["swarm.particles.per.cell.2D"] = 36
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1


# In[3]:


xmin, xmax = ndim(-1400 * u.kilometer), ndim(1400 * u.kilometer)
ymin, ymax = ndim(-700 * u.kilometer), ndim(0. * u.kilometer)
yint = 0.
 
dy = ndim(2.5 * u.kilometer)
dx = ndim(5.0 * u.kilometer)
xRes,yRes = int(np.around((xmax-xmin)/dx)),int(np.around((ymax-ymin)/dy))
yResa,yResb =int(np.around((ymax-yint)/dy)),int(np.around((yint-ymin)/dy))

#tRatio =  int(sys.argv[1])
#tRatio = 100
save_every = 1
use_fssa = False
if use_fssa:
    outputPath = "op_Ex_Crameri2012Case2_FreeSurf_ALE_withFSSA0.5_yres{:n}_uwg".format(yRes)

else:
    outputPath = "op_Ex_Crameri2012Case2_FreeSurf_ALE_noFSSA_yres{:n}_uwg".format(yRes)

Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -gravity))
Model.outputDir= outputPath
Model.minStrainRate = 1e-18 / u.second
print(xRes,yRes)


# In[4]:


hw = 150.* u.kilometer
hLith = -100.* u.kilometer
rPlume =  50.* u.kilometer  
x0Plume =  0.* u.kilometer 
y0Plume =  -400.* u.kilometer

PlumeShape = GEO.shapes.Disk(center=(x0Plume , y0Plume),radius=rPlume )

#Air_material = Model.add_material(name="Air_material", shape=GEO.shapes.Layer(top=Model.top, bottom=0.*u.km))
Li_material = Model.add_material(name="Li_material", shape=GEO.shapes.Layer(top=0.*u.km, bottom=hLith))
Ma_material = Model.add_material(name="Ma_material", shape=GEO.shapes.Layer(top=hLith, bottom=Model.bottom))
Pl_material = Model.add_material(name="Pl_material", shape=PlumeShape)


# In[5]:


npoints = xRes*2+1
coords = np.ndarray((npoints, 2))
coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
coords[:, 1] = GEO.nd(hLith) 
Model.add_passive_tracers(name="Interface", vertices=coords)

# coords = np.ndarray((npoints, 2))
# coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
# coords[:, 1] = 0.0
# #surf_tracers = Model.add_passive_tracers(name="Surface",vertices=coords)
# Model.add_passive_tracers(name="Surface", vertices=coords)


# In[6]:


# from underworld import visualisation as vis

# Fig = vis.Figure( figsize=(600,400),quality=2)
# Fig.Points(Model.swarm, Model.materialField, fn_size=2.0,colours='white green red purple blue yellow',discrete=True)
# Fig.show()

# figParticle.append (vis.objects.Points(swarm2,pointSize=5, colourBar=False))
# figParticle.append( vis.objects.Points(swarm, materialVariable, pointSize=2, colours='white green red purple blue yellow',discrete=True) )
# figParticle.show()

# Fig = vis.Figure(figsize=(500,500), title="Pressure Field (MPa)", quality=3)
# Fig.Surface(Model.mesh, GEO.dimensionalise(Model.pressureField, u.megapascal))
# Fig.show()

# Fig = vis.Figure(figsize=(500,500), title="Viscosity Field (Pa.s)", quality=3)
# Fig.Points(Model.swarm, 
#            GEO.dimensionalise(Model.viscosityField, u.pascal * u.second),
#            logScale=True,
#            fn_size=3.0)
# Fig.show()

# Fig = vis.Figure(figsize=(500,500), title="Density Field", quality=3)
# Fig.Points(Model.swarm, 
#            GEO.dimensionalise(Model.densityField, u.kg / u.m**3),
#            fn_size=3.0)
# Fig.show()


# In[7]:


#Air_material.density = 0. * u.kilogram / u.meter**3
Li_material.density = 3300 * u.kilogram / u.meter**3
Ma_material.density = 3300 * u.kilogram / u.meter**3
Pl_material.density = 3200 * u.kilogram / u.meter**3

#Air_material.viscosity = 1e19 * u.pascal * u.second
Li_material.viscosity = 1e23 * u.pascal * u.second
Ma_material.viscosity = 1e21 * u.pascal * u.second
Pl_material.viscosity = 1e20 * u.pascal * u.second

Model.set_kinematicBCs(left=[0., None],
                       right=[0., None],
                       top=[None, None],
                       bottom=[0.,0.])
#                       order_wall_conditions=[ "top", "left", "right", "bottom"])


Model.freeSurface = True

if use_fssa:
    Model._fssa_factor = 0.5
Model.init_model(pressure="lithostatic")

Model.solver.set_inner_method("mumps")

# In[8]:


if use_fssa:
    Model.run_for(10.1 * u.megayears, checkpoint_interval=1000000*u.year,dt=50000*u.year)
else:
    Model.run_for(10.1 * u.megayears, checkpoint_interval=1000000*u.year,dt=2500*u.year)


