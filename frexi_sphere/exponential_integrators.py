from firedrake import *
from abc import ABCMeta, abstractmethod
from rexi_coefficients import RexiCoefficients
from rexi import Rexi

class LinearExponentialIntegrator(object):
    """
    This class calculates exp(dtL)(U_in) using REXI for the linear
    shallow water equations.
    """

    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half):
        alpha, beta_re, beta_im = RexiCoefficients(h, M, 0, reduce_to_half)
        self.coefficients = alpha, beta_re
        self.rexi = Rexi(setup, dt, direct_solve)

    def apply(self, u_in, h_in, u_out, h_out):
        w = self.rexi.solve(u_in, h_in, self.coefficients)
        ur, hr, _, _ = w.split()
        u_out.assign(ur)
        h_out.assign(hr)


class NonlinearExponentialIntegrator(LinearExponentialIntegrator):
    """
    This is the base class for exponential integrators for the nonlinear
    shallow water equations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half):
        super(NonlinearExponentialIntegrator, self).__init__(setup, dt, direct_solve, h, M, reduce_to_half)
        self.dt = dt
        H = Constant(setup.params.H)
        V1 = setup.spaces['u']
        V2 = setup.spaces['h']
        self.u0 = Function(V1, name="u")
        self.h0 = Function(V2, name="h")

        W = MixedFunctionSpace((V1, V2))
        u, h = TrialFunctions(W)
        w, phi = TestFunctions(W)

        n = FacetNormal(setup.mesh)
        Upwind = 0.5*(sign(dot(self.u0, n))+1)
        if setup.outward_normals is not None:
            perp = lambda u: cross(setup.outward_normals, u)
            perp_u_upwind = lambda q: Upwind('+')*cross(setup.outward_normals('+'),q('+')) + Upwind('-')*cross(setup.outward_normals('-'),q('-'))
        else:
            perp = lambda u: as_vector([-u[1], u[0]])
            perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))
        un = 0.5*(dot(self.u0, n) + abs(dot(self.u0, n)))
        gradperp = lambda u: perp(grad(u))

        u_adv_term =(
            inner(gradperp(inner(w, perp(self.u0))), self.u0)*dx
            + inner(jump(inner(w, perp(self.u0)), n),
                    perp_u_upwind(self.u0))*dS
            +div(w)*(0.5*inner(self.u0, self.u0))*dx
        )
        h_cont_term = (
            +dot(grad(phi), self.u0)*(self.h0-H)*dx -
            dot(jump(phi), (un('+')*(self.h0('+')-H)
                            - un('-')*(self.h0('-')-H)))*dS
        )

        a = inner(w, u)*dx + phi*h*dx
        L = u_adv_term + h_cont_term
        self.Nw = Function(W)
        myprob = LinearVariationalProblem(a, L, self.Nw)
        self.nonlinear_solver = LinearVariationalSolver(myprob)

    @abstractmethod
    def apply(self, u_in, h_in, u_out, h_out):
        super(NonlinearExponentialIntegrator, self).apply(u_in, h_in, u_out, h_out)


class ETD1(NonlinearExponentialIntegrator):
    """
    This class implements the second order Exponential Time 
    Differencing (ETD) method described in equation 4 of 
    Cox and Matthews 2002.
    """
    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half):
        super(ETD1, self).__init__(setup, dt, direct_solve, h, M, reduce_to_half)
        alpha, beta_re, beta_im = RexiCoefficients(h, M, 1, reduce_to_half)
        self.phi1_coefficients = alpha, beta_re

    def apply(self, u_in, h_in, u_out, h_out):

        # calculate exp(dt L)U0
        super(ETD1, self).apply(u_in, h_in, u_out, h_out)

        # calculate N(U0)
        self.u0.assign(u_in)
        self.h0.assign(h_in)
        self.nonlinear_solver.solve()
        Nu, Nh = self.Nw.split()

        # calculate phi1(dt L)N(U0)
        w1 = self.rexi.solve(Nu, Nh, self.phi1_coefficients)
        ur, hr, _, _ = w1.split()

        u_out.assign(u_out + self.dt*ur)
        h_out.assign(h_out + self.dt*hr)


class ETD2RK(ETD1):
    """
    This class implements the second order Runge Kutta ETD
    method described in equations 20-22 of Cox and Matthews 2002:
    A_n = exp(dt L)U_n + dt phi_1(dt L)
    U_{n+1} = A_n + dt phi_2(dt L)(N(A_n) - N(U_n))
    """

    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half):
        super(ETD2RK, self).__init__(setup, dt, direct_solve, h, M, reduce_to_half)
        self.Nu = Function(setup.spaces["u"])
        self.Nh = Function(setup.spaces["h"])
        self.au = Function(setup.spaces["u"])
        self.ah = Function(setup.spaces["h"])
        alpha, beta_re, _ = RexiCoefficients(h, M, 2, reduce_to_half)
        self.phi2_coefficients = alpha, beta_re

    def apply(self, u_in, h_in, u_out, h_out):

        # calculate A
        super(ETD2RK, self).apply(u_in, h_in, self.au, self.ah)

        # save N(U)
        Nu, Nh = self.Nw.split()
        self.Nu.assign(-Nu)
        self.Nh.assign(-Nh)

        # calculate N(A)
        self.u0.assign(self.au)
        self.h0.assign(self.ah)
        self.nonlinear_solver.solve()
        Nau, Nah = self.Nw.split()

        # calculate phi2(dtL)(N(A) - N(U))        
        self.Nu += Nau
        self.Nh += Nah
        print self.Nu.dat.data.min(), self.Nu.dat.data.max()
        print self.Nh.dat.data.min(), self.Nh.dat.data.max()
        w2 = self.rexi.solve(self.Nu, self.Nh, self.phi2_coefficients)
        ur, hr, _, _ = w2.split()

        u_out.assign(self.au + self.dt*ur)
        h_out.assign(self.ah + self.dt*hr)


