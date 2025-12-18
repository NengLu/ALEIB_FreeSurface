import underworld3 as uw
from underworld3 import function
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local

from scipy.interpolate import interp1d
from enum import Enum
import numpy as np

class FreeSurfType(Enum):
    """
    free surface method type:

    FreeSurfType.CartesianALE     ALE in StructuredQuadBox 
    FreeSurfType.CartesianALEIB   ALE with internal boundary in StructuredQuadBox
    FreeSurfType.CartesianALEIBSP ALE with internal boundary and surface processes in StructuredQuadBox
    """
    CartesianALE = 0
    CartesianALEIB = 1
    CartesianALEIBSP = 2
    #AnnulusALE = 3
    #AnnulusALEIB = 4 

class FreeSurfaceProcessor_Cartesian(object): 
    def __init__(self,init_mesh,mesh,v,type = None,):
        """
        Parameters
        ----------
        _init_mesh : the original mesh
        mesh : the updating model mesh
        vel : the velocity field of the model
        dt : dt for advecting the surface
        """

        self.init_mesh = init_mesh
        self.Tmesh = uw.discretisation.MeshVariable("Tmesh", self.init_mesh, 1, degree=1)
        self.Bmesh = uw.discretisation.MeshVariable("Bmesh", self.init_mesh, 1, degree=1)
        self.mesh_solver = uw.systems.Poisson(self.init_mesh , u_Field=self.Tmesh)
        self.mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
        self.mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
        self.mesh_solver.f = 0.0
        self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Top")
        self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Bottom")

        if type == None:
            type = FreeSurfType.CartesianALE
        if not isinstance(type, FreeSurfType):
            raise ValueError("'type' must be an instance of 'FreeSurfType'")
        self.type = type 
        if self.type == FreeSurfType.CartesianALEIB or type == FreeSurfType.CartesianALEIBSP:
            self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Internal")
            self.interface = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Internal")
        elif self.type == FreeSurfType.CartesianALE:
            self.interface = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Top")

        #self.top  = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Top")
        #self.bottom= petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Bottom")

        self.mesh = mesh
        self.v = v
        
    def _solve_sle(self):
        self.mesh_solver.solve()                       
                 
    def _advect_surface(self): 
        with self.init_mesh.access(self.Bmesh):
            self.Bmesh.data[:, 0] = self.mesh.data[:, -1]
            #print("CPU.no: %d topsiez: %d \n" %(uw.mpi.rank,self.top.size))
            if self.interface.size > 0:
                if self.mesh.dim == 2:         
                    coords = self.mesh.data[self.interface]
                    vel = self.veldata[self.interface]
                    coords2 = coords + vel * self._dt
                    f = interp1d(coords2[:,0], coords2[:,1], kind='cubic', fill_value='extrapolate')
                    self.Bmesh.data[self.interface, 0] = f(coords[:,0])  
                else:
                    coords = self.mesh.data[self.interface]
                    vel = self.veldata[self.interface]
                    new_coords = coords + vel * self._dt
                    mesh_kdt = uw.kdtree.KDTree(coords[:,0:2].copy(order='C'))
                    mesh_kdt.build_index()
                    values = mesh_kdt.rbf_interpolator_local(new_coords[:,0:2].copy(order='C'),new_coords[:,-1][:, np.newaxis].copy(order='C'), self.mesh.dim+1)
                    del mesh_kdt
                    self.Bmesh.data[self.interface, 0] = values[:,0] 
        uw.mpi.barrier()
        self.init_mesh.update_lvec()
           
    def solve(self,dt):
        self._dt = dt

        #self.veldata = uw.function.evaluate(self.v.sym, self.mesh.data)
        # for v type = Q1
        with self.mesh.access():
            self.veldata = self.v.data

        self._advect_surface()
        self._solve_sle()
        with self.init_mesh.access():
            new_mesh_coords = self.init_mesh.data.copy()
            new_mesh_coords[:,-1] = self.Tmesh.data[:,0].copy()  
        return new_mesh_coords



class FreeSurfaceProcessor_Cartesian_SP(object): 
    def __init__(self,init_mesh,mesh,surfaceProcesses,type = None):
        """
        Parameters
        ----------
        _init_mesh : the original mesh
        mesh : the updating model mesh
        vel : the velocity field of the model
        dt : dt for advecting the surface
        """

        self.init_mesh = init_mesh
        self.Tmesh = uw.discretisation.MeshVariable("Tmesh", self.init_mesh, 1, degree=1)
        self.Bmesh = uw.discretisation.MeshVariable("Bmesh", self.init_mesh, 1, degree=1)
        self.mesh_solver = uw.systems.Poisson(self.init_mesh , u_Field=self.Tmesh)
        self.mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
        self.mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
        self.mesh_solver.f = 0.0
        self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Top")
        self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Bottom")

        if type == None:
            type = FreeSurfType.CartesianALE
        if not isinstance(type, FreeSurfType):
            raise ValueError("'type' must be an instance of 'FreeSurfType'")
        self.type = type 
        if self.type == FreeSurfType.CartesianALEIB or type == FreeSurfType.CartesianALEIBSP:
            self.mesh_solver.add_dirichlet_bc((self.Bmesh.sym[0],), "Internal")
            self.interface = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Internal")
        elif self.type == FreeSurfType.CartesianALE:
            self.interface = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Top")

        #self.top  = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Top")
        #self.bottom= petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Bottom")

        self.mesh = mesh
        #self.v = v

        #self._dt   = dt
        #self.top = topwall
        #self.bottom = botwall
        #self.internal = interwall 
        #self.dt_tectonic = dt
        self.surfaceProcesses = surfaceProcesses
        
    def _solve_sle(self):
        self.mesh_solver.solve()                       
                 
    # def _advect_surface(self): 
    #     with self.init_mesh.access(self.Bmesh):
    #         self.Bmesh.data[:, 0] = self.mesh.data[:, -1]
    #         #print("CPU.no: %d topsiez: %d \n" %(uw.mpi.rank,self.top.size))
    #         if self.interface.size > 0:
    #             if self.mesh.dim == 2:         
    #                 coords = self.mesh.data[self.interface]
    #                 vel = self.veldata[self.interface]
    #                 coords2 = coords + vel * self._dt
    #                 f = interp1d(coords2[:,0], coords2[:,1], kind='cubic', fill_value='extrapolate')
    #                 self.Bmesh.data[self.interface, 0] = f(coords[:,0])  
    #             else:
    #                 coords = self.mesh.data[self.interface]
    #                 vel = self.veldata[self.interface]
    #                 new_coords = coords + vel * self._dt
    #                 mesh_kdt = uw.kdtree.KDTree(coords[:,0:2].copy(order='C'))
    #                 mesh_kdt.build_index()
    #                 values = mesh_kdt.rbf_interpolator_local(new_coords[:,0:2].copy(order='C'),new_coords[:,-1][:, np.newaxis].copy(order='C'), self.mesh.dim+1)
    #                 del mesh_kdt
    #                 self.Bmesh.data[self.interface, 0] = values[:,0] 
    #     uw.mpi.barrier()
    #     self.init_mesh.update_lvec()

    def _advect_surface(self):
        with self.init_mesh.access(self.Bmesh):
            self.Bmesh.data[:, 0] = self.mesh.data[:, -1]
            self.surfaceProcesses.solve(self.dt_tectonic)
            if self.interface.size > 0: 
                coords = self.mesh.data[self.interface,:2]
                self.Bmesh.data[self.interface, 0] = uw.function.evaluate(self.surfaceProcesses.surfele.sym[0],coords)
        uw.mpi.barrier()
        self.init_mesh.update_lvec()
           
    def solve(self,dt):
        #self._dt = dt
        self.dt_tectonic = dt
        #self.veldata = uw.function.evaluate(self.v.sym, self.mesh.data)
        self._advect_surface()
        self._solve_sle()
        with self.init_mesh.access():
            new_mesh_coords = self.init_mesh.data.copy()
            new_mesh_coords[:,-1] = self.Tmesh.data[:,0].copy()  
        return new_mesh_coords