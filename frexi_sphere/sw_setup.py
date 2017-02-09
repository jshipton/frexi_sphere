from firedrake import *

class Configuration(object):

    def __init__(self, **kwargs):

        for name, value in kwargs.iteritems():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """Cause setting an unknown attribute to be an error"""
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
        object.__setattr__(self, name, value)


class ShallowWaterParameters(Configuration):

    """
    parameters for linear shallow water
    """
    Omega = 7.292e-5
    f = 1.e-4
    g = 9.8
    H = 1000.


class SetupShallowWater(object):

    """
    initial conditions and function spaces for linear shallow water
    """
    def __init__(self, mesh, family, degree, problem_name):
        self.mesh = mesh
        on_sphere = (mesh.geometric_dimension() == 3 and mesh.topological_dimension() == 2)
        if on_sphere:
            self.outward_normals = CellNormal(mesh)

        V1 = FunctionSpace(mesh, family, degree+1)
        V2 = FunctionSpace(mesh, "DG", degree)
        self.spaces = {'u': V1, 'h': V2}

        self.u0 = Function(self.spaces['u'])
        self.h0 = Function(self.spaces['h'])
        setup = getattr(self, problem_name)
        setup()

    def ex1(self):
        self.params = ShallowWaterParameters(f=1.0, g=1.0, H=1.0)
        x, y = SpatialCoordinate(self.mesh)
        self.u0.project(as_vector([cos(6*pi*x)*cos(4*pi*y) - 4*sin(6*pi*x)*sin(4*pi*y), cos(6*pi*x)*cos(6*pi*y)]))
        self.h0.interpolate(sin(6*pi*x)*cos(4*pi*y) - 0.2*cos(4*pi*x)*sin(2*pi*y))
    def wave_scenario(self):
        self.params = ShallowWaterParameters(f=1.0, g=1.0, H=1.0)
        x, y = SpatialCoordinate(self.mesh)
        self.u0.project(as_vector([cos(8*pi*x)*cos(2*pi*y), cos(4*pi*x)*cos(4*pi*y)]))
        self.h0.interpolate(sin(4*pi*x)*cos(2*pi*y) - 0.2*cos(4*pi*x)*sin(4*pi*y))

    def gaussian_scenario(self):
        self.params = ShallowWaterParameters(f=1.0, g=1.0, H=1.0)
        x, y = SpatialCoordinate(self.mesh)
        self.h0.interpolate(exp(-50*((x-0.5)**2 + (y-0.5)**2)))


    def polar_wave(self):
        self.params = ShallowWaterParameters()
        x = SpatialCoordinate(self.mesh)
        R = self.mesh._icosahedral_sphere
        Dexpr = Expression("R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0)) < rc ? 50.*h0*(1 + cos(pi*R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0))/rc)) : 0.0", R=R, rc=R/3., h0=self.params.H, x0=0.0, x1=0.0, x2=R)
        self.h0.interpolate(Dexpr)

    def linear_w2(self):
        self.params = ShallowWaterParameters(H=2000.)
        x = SpatialCoordinate(self.mesh)
        R = self.mesh._icosahedral_sphere
        Omega = self.params.Omega
        fexpr = 2*Omega*x[2]/R
        V = FunctionSpace(self.mesh, "CG", 1)
        self.params.f = Function(V).interpolate(fexpr)
        day = 24.*60.*60.
        u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
        u_max = Constant(u_0)
        uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
        g = Constant(self.params.g)
        Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
        self.h0.interpolate(Dexpr)
        self.u0.project(uexpr)

    def w2(self):
        self.params = ShallowWaterParameters(H=2996.942)
        x = SpatialCoordinate(self.mesh)
        R = self.mesh._icosahedral_sphere
        Omega = self.params.Omega
        fexpr = 2*Omega*x[2]/R
        V = FunctionSpace(self.mesh, "CG", 1)
        self.params.f = Function(V).interpolate(fexpr)
        day = 24.*60.*60.
        u_0 = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
        u_max = Constant(u_0)
        uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
        g = Constant(self.params.g)
        h0 = Constant(self.params.H)
        Dexpr = h0 - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
        self.h0.interpolate(Dexpr)
        self.u0.project(uexpr)

    def w5(self):
        self.params = ShallowWaterParameters(H=5960.)
        x = SpatialCoordinate(self.mesh)
        R = self.mesh._icosahedral_sphere
        Omega = self.params.Omega
        fexpr = 2*Omega*x[2]/R
        V = FunctionSpace(self.mesh, "CG", 1)
        self.params.f = Function(V).interpolate(fexpr)
        day = 24.*60.*60.
        u_0 = 20.
        u_max = Constant(u_0)
        uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
        g = Constant(self.params.g)
        h0 = Constant(self.params.H)
        Dexpr = Expression("h0 - ((R0 * Omega * u0 + pow(u0,2)/2.0)*(x[2]*x[2]/(R0*R0)))/g - (2000 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2)+pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0)))", h0=h0, R0=R, Omega=Omega, u0=20.0, g=g)
        self.h0.interpolate(Dexpr)
        self.u0.project(uexpr)
