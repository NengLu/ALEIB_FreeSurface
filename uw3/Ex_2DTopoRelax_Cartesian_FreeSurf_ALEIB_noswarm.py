#!/usr/bin/env python
# coding: utf-8

# In[1]:


import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3 import function
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local

import numpy as np
import sympy
import os
from datetime import datetime
import sys
#import matplotlib.pyplot as plt


# In[2]:


u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

# scaling 3: vel
half_rate = 1.0 * u.centimeter / u.year
model_length = 500. * u.kilometer
gravity = 9.81 * u.meter / u.second**2
bodyforce = 3300 * u.kilogram / u.metre**3 *gravity 

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM


xmin, xmax = ndim(-250 * u.kilometer), ndim(250 * u.kilometer)
ymin, ymax = ndim(-500 * u.kilometer), ndim(100 * u.kilometer)
yint = 0.
# xres, yres = 50,60
# dy = (ymax-ymin)/yres
# dx = (xmax-xmin)/xres

dy = ndim(500 * u.kilometer)/50
dx = dy
xres,yres = int((xmax-xmin)/dx),int((ymax-ymin)/dy)
yresa,yresb =int((ymax-yint)/dy),int((yint-ymin)/dy)


tRatio =  int(sys.argv[1])
#tRatio = 100
save_every = 1
use_fssa = False
if use_fssa:
    outputPath = "op_Ex_2DTopoRelax_Cartesain_FreeSurf_ALEIB_withFSSA0.5_yres{:n}_tRatio{:n}_noSwarm/".format(yres,tRatio)
else:
    outputPath = "op_Ex_2DTopoRelax_Cartesain_FreeSurf_ALEIB_noFSSA_yres{:n}_tRatio{:n}_noSwarm/".format(yres,tRatio)
if uw.mpi.rank == 0:
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+ i)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

mesh = uw.meshing.BoxInternalBoundary(elementRes=(xres,yres),zelementRes=(yresa,yresb),minCoords=(xmin,ymin),maxCoords=(xmax, ymax),zintCoord=yint,degree=1,qdegree=2)
init_mesh = mesh = uw.meshing.BoxInternalBoundary(elementRes=(xres,yres),zelementRes=(yresa,yresb),minCoords=(xmin,ymin),maxCoords=(xmax, ymax),zintCoord=yint,degree=1,qdegree=2)
                                                  
# # dq2dq1
# v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
# p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# q1dq0
v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1,continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0,continuous=False)
timeField     = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)

botwall = petsc_dm_find_labeled_points_local(mesh.dm,"Bottom")
topwall = petsc_dm_find_labeled_points_local(mesh.dm,"Top")
interwall = petsc_dm_find_labeled_points_local(mesh.dm,'Internal')


# In[3]:


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
deform_fn = w_m * sympy.cos(2.*np.pi*(mesh.X[0])/Lambda)

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


# In[4]:

# In[5]:


R0 = uw.discretisation.MeshVariable("r_0", mesh, vtype=uw.VarType.SCALAR, degree=2, continuous=False)

with mesh.access(R0):
    R0.data[:,0] = uw.function.evaluate(mesh.X[1], R0.coords)
    #print(R0.data[:,0])


viscM = ndim(1e21 * u.pascal * u.second)
densityM = ndim(3300 * u.kilogram / u.metre**3)

viscA = ndim(1e18 * u.pascal * u.second)
densityA = ndim(0. * u.kilogram / u.metre**3)

density_fn = sympy.Piecewise((densityA,R0.sym[0]>0),
                             (densityM, True))
visc_fn = sympy.Piecewise((viscA,R0.sym[0]>0),
                          (viscM, True))


# In[6]:


M = uw.discretisation.MeshVariable("M", mesh, 1, degree=1)

diffuser = uw.systems.Poisson(mesh, M)
diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
diffuser.constitutive_model.Parameters.diffusivity = 1.

diffuser.add_essential_bc((ymax,), "Top")
diffuser.add_essential_bc((deform_fn,), "Internal")
diffuser.add_essential_bc((ymin,),"Bottom")
diffuser.solve()

displacementy = uw.function.evaluate(M.sym[0], mesh.data)
displacement = np.zeros([displacementy.shape[0],2])
displacement[:,0] = mesh.data[:,0]
displacement[:,1]= displacementy
mesh.deform_mesh(displacement)


# In[8]:


ND_gravity = g

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.shear_viscosity_0
stokes.add_essential_bc((0.0,None), "Left")
stokes.add_essential_bc((0.0,None), "Right")
stokes.add_essential_bc((0.0,0.0), "Bottom")
stokes.add_essential_bc((None,0.0), "Top")

# if uw.mpi.size == 1:
#     stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = 1.0e-6
stokes.petsc_options["ksp_rtol"] = 1.0e-6
stokes.petsc_options["ksp_atol"] = 1.0e-6
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None


if use_fssa:
    theta = 0.5*drho*ND_gravity*dt_set
    FSSA_traction = theta*mesh.Gamma.dot(v.sym) * mesh.Gamma
    stokes.add_natural_bc(FSSA_traction, mesh.boundaries.Internal.name)


# In[9]:


def _adjust_time_units(val):
    """ Adjust the units used depending on the value """
    if isinstance(val, u.Quantity):
        mag = val.to(u.years).magnitude
    else:
        val = dim(val, u.years)
        mag = val.magnitude
    exponent = int("{0:.3E}".format(mag).split("E")[-1])

    if exponent >= 9:
        units = u.gigayear
    elif exponent >= 6:
        units = u.megayear
    elif exponent >= 3:
        units = u.kiloyears
    elif exponent >= 0:
        units = u.years
    elif exponent > -3:
        units = u.days
    elif exponent > -5:
        units = u.hours
    elif exponent > -7:
        units = u.minutes
    else:
        units = u.seconds
    return val.to(units)


# In[10]:


from freesurface import FreeSurfaceProcessor_Cartesian
from freesurface import FreeSurfType

freesuface = FreeSurfaceProcessor_Cartesian(init_mesh,mesh,v,type=FreeSurfType.CartesianALEIB)


# In[11]:


step      = 0
max_steps = 5
time      = 0
dt        = 0

while time < max_time+dt_set:   
#while step < max_steps:
    
    if uw.mpi.rank == 0:
        string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
        step, _adjust_time_units(time),
        _adjust_time_units(dt),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.stdout.write(string)
        sys.stdout.flush()

    stokes.solve(zero_init_guess=False,_force_setup=True)
    #stokes.solve(zero_init_guess=False)

    if step%save_every ==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data:')
        with mesh.access(timeField):
            timeField.data[:,0] = dim(time, u.megayear).m
        mesh.petsc_save_checkpoint(meshVars=[v, p, timeField], index=step, outputPath=outputPath)
        #swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputPath) 
        #swarm.write_timestep('', "swarm", swarmVars=[material],outputPath=outputPath,index=step,force_sequential=True)

    dt_solver = stokes.estimate_dt()
    dt = min(dt_solver,dt_set)

    #swarm.advection(V_fn=stokes.u.sym, delta_t=dt,order=1)

    new_mesh_coords=freesuface.solve(dt)
    mesh.deform_mesh(new_mesh_coords)

    # if uw.mpi.rank == 0:
    #     print(f'\nrepopulate start:')
    # pop_control.repopulate(mesh,material,maxppc=n_pincell,minppc=minppc)
    # if uw.mpi.rank == 0:
    #     print(f'\nrepopulate end:')

    step += 1
    time += dt