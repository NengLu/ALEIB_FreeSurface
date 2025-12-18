#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld as uw
import math
from underworld import function as fn
import numpy as np
import os

from _freesurface_ALEIB import FreeSurfaceProcessor_ALEIB
from underworld import UWGeodynamics as GEO
u = GEO.UnitRegistry
ndim = GEO.non_dimensionalise
dimen = GEO.dimensionalise

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size

half_rate = 1. * u.centimeter / u.year
model_length = 100. * u.kilometer
bodyforce = 3300 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2
gravity = 9.81 * u.meter / u.second**2


KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

scaling_coefficients = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM


# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 5e-3
GEO.rcParams['initial.nonlinear.max.iterations'] = 200
GEO.rcParams["nonlinear.tolerance"] = 5e-3
GEO.rcParams['nonlinear.max.iterations'] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 20
GEO.rcParams["swarm.particles.per.cell.2D"] = 20
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1

yRes = 160
xRes = 192
npoints = 50

xmin, xmax = ndim(0.0 * u.kilometer), ndim(900 * u.kilometer)
ymin, ymax = ndim(-600 * u.kilometer), ndim(150 * u.kilometer)

dx = (xmax-xmin)/xRes
dy = (ymax-ymin)/yRes


use_fssa = False
if use_fssa:
    outputPath = "op_delamination_Fsurf_ALEIB_withFSSA0.5_yres{:n}_uwg".format(yRes)
else:
    outputPath = "op_delamination_Fsurf_ALEIB_noFSSA_yres{:n}_uwg".format(yRes)

Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -gravity))
Model.outputDir= outputPath
Model.minStrainRate = 1e-18 / u.second
print(xRes,yRes)


# In[2]:


# Thickness of the dense material
L = 1.0
# Thickness of decollement layer
Mlcrust = 0.20
# Thickness of upper crust.
#     If set to zero, the upper BC will be set to no-slip,
#     representing an infinitely strong upper crust.
Mcrust = 0.20
M = Mlcrust + Mcrust
D = 1.0
etaLc = 1e-2

ymax0 = 0.
Ducrust = ymax0 - Mcrust*L
Dlcrust = ymax0 - M*L

def InterfaceY(x,arrN,arrA,D):
    if x > 0.5*(xmax-xmin):
        thinned = True
    else:
        thinned = False

    pertX = 0.
    if not thinned or D < 1:
        for i in range(len(arrN)):
            pertX += arrA[i] * np.cos(2.*np.pi * arrN[i] / (xmax-xmin) * x )

    interface = ymax0 - L*(M+1.) + pertX

    if thinned:
        interface += D*L

    if interface > ymax0 - M*L:
        interface = ymax0 - M*L

    return interface  


AirShape = GEO.shapes.Layer(top=Model.top, bottom=0.)
iAsthenShape = GEO.shapes.Layer(top=Dlcrust, bottom=Model.bottom)
iCrustShape = GEO.shapes.Layer(top=0., bottom=Ducrust )
iLowercrustShape = GEO.shapes.Layer(top=Ducrust, bottom=Dlcrust)
iDB_polygon = np.vstack(([xmin,-1.4],[4.5,-1.4],[4.5,-0.4],[xmin,-0.4]))
iDBShape = GEO.shapes.Polygon(iDB_polygon)


# In[4]:


Air    = Model.add_material(name="Air", shape=AirShape)
iAsthen  = Model.add_material(name="iAsthen", shape=iAsthenShape)
iCrust = Model.add_material(name="iCrust ", shape=iCrustShape)
iLowercrust  = Model.add_material(name="iLowercrust", shape=iLowercrustShape)
iDB = Model.add_material(name="iDB", shape=iDBShape)


# In[5]:


# from underworld import visualisation as vis

# Fig = vis.Figure( figsize=(600,400),quality=2)
# Fig.Points(Model.swarm, Model.materialField, fn_size=2.0,discrete=True) #,colours='white green red purple blue yellow',discrete=True)
# Fig.show()


# In[6]:


Air.density = 0.   
iAsthen.density =  ndim(3250 * u.kilogram / u.metre**3)
iCrust.density =   ndim(2800 * u.kilogram / u.metre**3)
iLowercrust.density =   ndim(3300 * u.kilogram / u.metre**3)
iDB.density =   ndim(3300 * u.kilogram / u.metre**3)

DBEta = ndim(1e21  * u.pascal * u.second)
Air.viscosity = ndim(1e19  * u.pascal * u.second)  
iAsthen.viscosity =  1e-3 * DBEta  
iCrust.viscosity =  DBEta * 1e2  
iLowercrust.viscosity =  DBEta * etaLc 
iDB.viscosity =   DBEta


# In[7]:


Model.set_velocityBCs(left=[0.,None],right=[0,None],bottom=[None,0.], top=[None, 0.])


# In[8]:


Model.init_model(pressure='lithostatic',temperature=None)


# In[ ]:


# if use_fssa:
#    # Model.run_for(4.1 * u.megayears, checkpoint_interval=0.1 * u.megayears,dt=50000*u.year)
#     Model.run_for(maxTime, checkpoint_interval=checkpoint_interval,dt=50000*u.year)
# else:
#     Model.run_for(maxTime, checkpoint_interval=checkpoint_interval,dt=dt_set)


# # In[ ]:


D_ib = 0.0 
axis_ib = np.where((Model.mesh.data[:,1]<=D_ib+dy/4)&(Model.mesh.data[:,1]>=D_ib-dy/4))
Sets_ib = Model.mesh.specialSets["Empty"]
for index in axis_ib:
    Sets_ib.add(index)

Model._freeSurface = True
Model.inter_wall = Sets_ib
Model._freeSurface = FreeSurfaceProcessor_ALEIB(Model)

Model.run_for(4.1e5*u.year, checkpoint_interval=1e4*u.year,dt=2500*u.year)


# In[ ]:


#Model.run_for(nstep=10, checkpoint_interval=1)


# In[ ]:


# Fig = vis.Figure( figsize=(600,400),quality=2)
# Fig.Points(Model.swarm, Model.materialField, fn_size=2.0,discrete=True) #,colours='white green red purple blue yellow',discrete=True)
# Fig.VectorArrows(Model.mesh, Model.velocityField) 
# Fig.show()


# In[ ]:




