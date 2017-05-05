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
        else:
            self.outward_normals = None

        V1 = FunctionSpace(mesh, family, degree+1)
        V2 = FunctionSpace(mesh, "DG", degree)
        self.spaces = {'u': V1, 'h': V2}

        self.u0 = Function(self.spaces['u'])
        self.h0 = Function(self.spaces['h'])
        self.ics = getattr(self, problem_name)
        self.ics()

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

    def linear_w2(self, topography=False):
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
        if topography:
            bexpr = Expression("700 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2)+pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=R)
            self.b = Function(self.h0.function_space())
            self.b.interpolate(bexpr)

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
        bexpr = Expression("2000 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2)+pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=R)
        self.b = Function(self.h0.function_space())
        self.b.interpolate(bexpr)

    def merging_vortices(self):
        self.params = ShallowWaterParameters(H=1., f=1., g=1.)
        f = self.params.f
        g = self.params.g
        H = self.params.H
        k = Constant(0.003)
        a = Constant(.1)
        x0 = Constant(0.425)
        y0 = Constant(0.5)
        x1 = Constant(0.575)
        y1 = Constant(0.5)
        x, y = SpatialCoordinate(self.mesh)
        psi_expr = (k/a)*(exp(-((x-x0)**2+(y-y0)**2)/(a**2)) + exp(-((x-x1)**2+(y-y1)**2)/(a**2)))
        V0 = FunctionSpace(self.mesh, "CG", 2)
        psi = Function(V0, name='psi').interpolate(psi_expr)
        self.u0.project(perp(grad(psi)))
        V1 = self.u0.function_space()
        V2 = self.h0.function_space()
        W = V1*V2
        F = Function(W)
        v = Function(V1).project(f*perp(self.u0))
        z, D = TrialFunctions(W)
        w, phi = TestFunctions(W)
        a = inner(w, z)*dx + div(w)*g*D*dx + phi*div(z)*dx
        L = -phi*div(v)*dx
        params = {'ksp_type':'gmres',
                  'pc_type': 'fieldsplit',
                  'pc_fieldsplit_type': 'schur',
                  'pc_fieldsplit_schur_fact_type': 'full',
                  'pc_fieldsplit_schur_precondition': 'selfp',
                  'fieldsplit_1_ksp_type': 'preonly',
                  'fieldsplit_1_pc_type': 'gamg',
                  'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                  'fieldsplit_1_mg_levels_sub_pc_type': 'ilu',
                  'fieldsplit_0_ksp_type': 'richardson',
                  'fieldsplit_0_ksp_max_it': 4,
                  'ksp_atol': 1.e-08,
                  'ksp_rtol': 1.e-08}
        solve(a==L, F, solver_parameters=params)
        z, D = F.split()
        self.h0.assign(H+D)
