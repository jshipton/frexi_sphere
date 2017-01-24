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
    f = 1.e-4
    g = 9.8
    H = 1000.


class SetupShallowWater(object):

    """
    initial conditions for linear shallow water
    """
    def __init__(self, mesh, family, degree, problem_name):
        self.mesh = mesh
        self.V1 = FunctionSpace(mesh, family, degree+1)
        self.V2 = FunctionSpace(mesh, "DG", degree)
        self.u0 = Function(self.V1)
        self.h0 = Function(self.V2)
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
