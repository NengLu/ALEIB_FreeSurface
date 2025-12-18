#!/usr/bin/env python
# coding: utf-8

# In[1]:


# modified from 2D Subduction from Crameri et al 2017


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
#model_length = 2890 * u.km
gravity = 10.0 * u.meter / u.second**2 
ref_density   = 3300 * u.kilogram / u.meter**3
bodyforce = ref_density * gravity
refViscosity = 1e21 * u.pascal * u.seconds 

refTempSurf = 300 *  u.degK
refTempLAB = 1600 *  u.degree_Kelvin
refTempBot = 1600 *  u.degree_Kelvin

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2
KT = 1600 *  u.degree_Kelvin

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
GEO.scaling_coefficients["[temperature]"] = KT

# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-2
GEO.rcParams['initial.nonlinear.max.iterations'] = 50
GEO.rcParams["nonlinear.tolerance"] = 1e-2
GEO.rcParams['nonlinear.max.iterations'] = 20
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 16
GEO.rcParams["swarm.particles.per.cell.2D"] = 16
GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True
GEO.rcParams["popcontrol.split.threshold"] = 0.1


xmin, xmax = ndim(-1500 * u.kilometer), ndim(1500 * u.kilometer)
ymin, ymax = ndim(-800 * u.kilometer), ndim(200 * u.kilometer)
yint = 0.

boxHalf = (xmax-xmin)*0.5
#hpc
dy = ndim(5.0 * u.kilometer)
dx = ndim(5.0 * u.kilometer)

#local
# dy = ndim(10.0 * u.kilometer)
# dx = ndim(10.0 * u.kilometer)

xRes,yRes = int(np.around((xmax-xmin)/dx)),int(np.around((ymax-ymin)/dy))
yResa,yResb =int(np.around((ymax-yint)/dy)),int(np.around((yint-ymin)/dy))

use_fssa = False
if use_fssa:
    outputPath = "op_Ex_Subduction_FreeSurf_Eulerian_withFSSA0.5_yres{:n}_uwg".format(yRes)
else:
    outputPath = "op_Ex_Subduction_FreeSurf_Eulerian_noFSSA_yres{:n}_uwg".format(yRes)

Model = GEO.Model(elementRes=(xRes, yRes),
                  minCoord=(xmin,ymin),
                  maxCoord=(xmax, ymax),
                  gravity=(0.0, -gravity))
Model.outputDir= outputPath
Model.minStrainRate = 1e-18 / u.second

WBL0 = 100/1000 
thcrust = 7.5/1000
slab_theta = 30
slab_length = 500/1000


depthFn = 0.-Model.y
WBLFn = WBL0 * fn.math.sqrt(fn.math.abs(Model.x-boxHalf)/boxHalf) # platethickness 
platethickness = fn.branching.conditional([(Model.x > 0., WBLFn ),
                                          (True,         WBL0)]) 

def thickness_x(x,th0):
    return th0 * np.sqrt(np.abs(x-boxHalf)/boxHalf) # platethickness

def erf_custom(x):
    # Define the number of steps for integration
    n = 100000  # More steps for better accuracy
    step = x / n
    integral = 0.0

    # Trapezoidal integration
    for i in range(n):
        t = i * step
        integral += np.exp(-t**2) * step

    return (2 / np.sqrt(np.pi)) * integral

AirShape = GEO.shapes.Layer(top=Model.top, bottom=0.)
MantleShape = GEO.shapes.Layer(top=Model.top, bottom=Model.bottom)

slab_sintheta = np.sin(np.radians(slab_theta))
slab_costheta = np.cos(np.radians(slab_theta))

def subpolygon_fn(sub_th,sub_theta,sub_length,xRes):
    slab_sintheta = np.sin(np.radians(sub_theta))
    slab_costheta = np.cos(np.radians(sub_theta))
    slab_x0 = -sub_length*slab_costheta 
    slab_y0 = -sub_length*slab_sintheta 
    slab_x1 = slab_x0 + sub_th*slab_sintheta
    slab_y1 = slab_y0 - sub_th*slab_costheta
    slab_x2,slab_y2 = 0.,-sub_th
    slab_x3,slab_y3 = 0.,0. 
    npoints = xRes+1
    slab_bot = np.zeros([npoints,2]) 
    slab_bot_x = np.linspace(0.,xmax,npoints)
    slab_bot[:,0] = slab_bot_x
    slab_bot[:,1] = -thickness_x(slab_bot_x,sub_th) 
    return np.vstack(([slab_x0,slab_y0],[slab_x1,slab_y1], [slab_x2,slab_y2], slab_bot,[slab_x3,slab_y3]))

def onlyslabpolygon_fn(sub_th,sub_theta,sub_length,xRes):
    slab_sintheta = np.sin(np.radians(sub_theta))
    slab_costheta = np.cos(np.radians(sub_theta))
    slab_x0 = -sub_length*slab_costheta 
    slab_y0 = -sub_length*slab_sintheta 
    slab_x1 = slab_x0 + sub_th*slab_sintheta
    slab_y1 = slab_y0 - sub_th*slab_costheta
    slab_x2,slab_y2 = 0.,-sub_th
    slab_x3,slab_y3 = 0.,0. 
    return np.vstack(([slab_x0,slab_y0],[slab_x1,slab_y1], [slab_x2,slab_y2],[slab_x3,slab_y3]))

sub_polygon = subpolygon_fn(WBL0,slab_theta,slab_length,xRes)
slabshape_polygon = np.vstack(([xmin,-WBL0],[-WBL0/np.tan(np.radians(slab_theta)),-WBL0],sub_polygon,[xmin,0]))

crust_polygon = subpolygon_fn(thcrust,slab_theta,slab_length,xRes)

SlabShape = GEO.shapes.Polygon(slabshape_polygon)
CrustShape = GEO.shapes.Polygon(crust_polygon)


Mantle = Model.add_material(name="Mantle", shape=MantleShape)
Slab   = Model.add_material(name="Slab", shape=SlabShape)
Crust  = Model.add_material(name="Crust", shape=CrustShape)
Air    = Model.add_material(name="Air", shape=AirShape)

npoints = xRes*2+1

coords = np.zeros([npoints,2])
coords[:,0] = np.linspace(xmin,xmax,npoints)
coords[:,1] = 0.

surf_tracers = Model.add_passive_tracers(name="Surface", vertices=coords) 

class DruckerPrager_Byerlee(object):
    def __init__(self, name=None, cohesion=None, frictionCoefficient=None):
        """
        using a Drucker-Prager yield criterion with the pressure-dependent yield stress sigma_y based on Byerlee’s law

        Drucker Prager yield Rheology.
         The pressure-dependent Drucker-Yield criterion is defined as follow:
         .. math::
            $ \sigma_{y,brittle} = C  + \mu P$

        Parameters
        ----------
            cohesion :
                Cohesion for the pristine material(initial cohesion)
            frictionCoefficient :
                friction angle for a pristine material
            ductileYieldStress :    
        Returns
        -------
        An UWGeodynamics DruckerPrager class
        """
        self.name = name
        self._cohesion = cohesion
        self._frictionCoefficient = frictionCoefficient

        self.plasticStrain = None
        self.pressureField = Model.pressureField

    @property
    def cohesion(self):
        return self._cohesion

    @cohesion.setter
    def cohesion(self, value):
        self._cohesion = value

    @property
    def frictionCoefficient(self):
        return self._frictionCoefficient

    @frictionCoefficient.setter
    def frictionCoefficient(self, value):
        self._frictionCoefficient = value

    def _frictionFn(self):
        friction = fn.misc.constant(self.frictionCoefficient)
        return friction

    def _cohesionFn(self):
        cohesion = fn.misc.constant(ndim(self.cohesion))
        return cohesion

    def _get_yieldStress2D(self):
        f = self._frictionFn()
        C = self._cohesionFn()
        P = self.pressureField
        self.yieldStress = C + P * f 
        return self.yieldStress

    def _get_yieldStress3D(self):
        print("no settings")
        return

#rh = GEO.ViscousCreepRegistry()

Model.maxViscosity = refViscosity*1e5
Model.minViscosity = refViscosity*1e-4


ref_viscosity_A = refViscosity/ np.exp((240 *1e3/(1600*8.314)))  
ductileViscosity = GEO.ViscousCreep(name='ductile',
                                 preExponentialFactor=1./ref_viscosity_A,
                                 stressExponent=1.0,
                                 activationVolume=0.,activationEnergy=240 * u.kilojoules/u.mole,
                                 f=2.0)  # Crameri et al., 2017

Mantle.viscosity = 1e21* u.pascal * u.seconds  #refViscosity  # Isoviscous
Slab.viscosity   = ductileViscosity  
Crust.viscosity  = ductileViscosity 
Air.viscosity = 1e19 * u.pascal * u.seconds

mplasticity  = DruckerPrager_Byerlee(name='mantlepl',cohesion=10. * u.megapascal,frictionCoefficient = 0.25)
cplasticity  = DruckerPrager_Byerlee(name='crustpl',cohesion=10. * u.megapascal,frictionCoefficient = 0.001)

Mantle.plasticity = mplasticity
Slab.plasticity  = mplasticity
Crust.plasticity = cplasticity
Slab.stressLimiter = ndim(600. * u.megapascal)
Crust.stressLimiter = ndim(600. * u.megapascal)


r_tE = 3e-5 / u.kelvin
r_tem = 300 * u.kelvin

alldensity  = GEO.LinearDensity(3300. * u.kilogram / u.metre**3,thermalExpansivity = r_tE, reference_temperature = r_tem )     
alldensity.temperatureField = Model.temperature

Mantle.density = alldensity 
Slab.density   = alldensity    
Crust.density  = alldensity  
Air.density = 0.

Model.diffusivity = 1.0e-6 * u.metre**2 / u.second
Model.capacity    = 1200. * u.joule / (u.kelvin * u.kilogram)

Mantle.diffusivity = 1.0e-6 * u.metre**2 / u.second
Slab.diffusivity = 1.0e-6 * u.metre**2 / u.second 
Crust.diffusivity = 1.0e-6 * u.metre**2 / u.second
Air.diffusivity = 20.0e-6 * u.metre**2 / u.second 


Mantle.capacity  = 1200. * u.joule / (u.kelvin * u.kilogram)
Slab.capacity    = 1200. * u.joule / (u.kelvin * u.kilogram)
Crust.capacity   = 1200. * u.joule / (u.kelvin * u.kilogram)
Air.capacity = 100. * u.joule / (u.kelvin * u.kilogram)


Mantle.radiogenicHeatProd  = 0.022 * u.microwatt / u.meter**3
Slab.radiogenicHeatProd    = 0.2 * u.microwatt / u.meter**3 
Crust.radiogenicHeatProd   = 0.2 * u.microwatt / u.meter**3 
Air.radiogenicHeatProd   =  0.

Temp_surf = 300/1600
Temp_bot = 1600/1600
Temp0 = 1600/1600
Temp_mantle = Temp0 

halfSpaceTemp = (Temp_bot-Temp_surf)*fn.math.erf((depthFn)/platethickness)+Temp_surf
geotherm_fn = fn.branching.conditional([(depthFn <= 0.,Temp_surf),
                                        (True,  halfSpaceTemp)])
Model._init_temperature_variables()
Model.temperature.data[...] = 0.
Model.temperature.data[...] = geotherm_fn.evaluate(Model.mesh)

onlyslab_polygon = onlyslabpolygon_fn(WBL0,slab_theta,slab_length,xRes)

slab_index = GEO.shapes.Polygon(onlyslab_polygon).evaluate(Model.mesh.data)[:,0]
slab_coords = Model.mesh.data[slab_index]

temth = np.zeros(slab_coords.shape[0])

for index in range(slab_coords.shape[0]):
    coord = slab_coords[index][:]
    coordx = coord[0]
    coordy = coord[1]
    temth[index] = coordx*slab_sintheta-coordy*slab_costheta

Model.temperature.data[slab_index,0]= (Temp_bot-Temp_surf)*erf_custom(temth/WBL0)+Temp_surf 

Model.set_velocityBCs(left=[0.,None],right=[0,None],bottom=[None,0.], top=[None, None])
Model.set_temperatureBCs(top=refTempSurf, materials=[(Air, Temp_surf)]) 

Model.pressureField.data[...] = 0.
Model.init_model(pressure='lithostatic',temperature=None,defaultStrainRate=1e-15*u.second)

Model.solver.set_inner_method("mumps")
#Model.solver.set_penalty(1e6)

maxTime = 2.01*u.megayear
dt_set = 5.0*u.kiloyear 
checkpoint_interval = 10*u.kiloyear

if use_fssa:
    Model.run_for(maxTime, checkpoint_interval=checkpoint_interval,dt=50000*u.year)
else:
    Model.run_for(maxTime, checkpoint_interval=checkpoint_interval,dt=dt_set)