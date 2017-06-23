from firedrake import *
from abc import ABCMeta, abstractmethod
from rexi_coefficients import RexiCoefficients
from rexi import Rexi
import numpy as np

class LinearExponentialIntegrator(object):
    """
    This class calculates exp(dtL)(U_in) using REXI for the linear
    shallow water equations.
    """

    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half):
        self.dt = dt
        alpha, beta_re, beta_im = RexiCoefficients(h, M, 0, reduce_to_half)
        self.coefficients = alpha, beta_re
        self.rexi = Rexi(setup, direct_solve, self.coefficients)

    def apply(self, u_in, h_in, u_out, h_out, dt=None):
        if dt is None:
            dt = self.dt
        w = self.rexi.solve(u_in, h_in, dt)
        ur, hr, _, _ = w.split()
        u_out.assign(ur)
        h_out.assign(hr)


class NonlinearExponentialIntegrator(LinearExponentialIntegrator):
    """
    This is the base class for exponential integrators for the nonlinear
    shallow water equations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half, nonlinear=True):
        super(NonlinearExponentialIntegrator, self).__init__(setup, dt, direct_solve, h, M, reduce_to_half)
        H = Constant(setup.params.H)
        V1 = setup.spaces['u']
        V2 = setup.spaces['h']
        self.u0 = Function(V1, name="u")
        self.h0 = Function(V2, name="h")

        W = MixedFunctionSpace((V1, V2))
        u, h = TrialFunctions(W)
        w, phi = TestFunctions(W)
        a = inner(w, u)*dx + phi*h*dx

        if nonlinear:
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

            L = u_adv_term + h_cont_term

        if hasattr(setup, 'b'):
            b_term = setup.params.g*(div(w)*setup.b*dx - inner(jump(w, n), un('+')*setup.b('+') - un('-')*setup.b('-'))*dS)
            if nonlinear:
                L += b_term
            else:
                L = b_term

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
        w1 = self.rexi.solve(Nu, Nh, dt, self.phi1_coefficients)
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
        w2 = self.rexi.solve(self.Nu, self.Nh, dt, self.phi2_coefficients)
        ur, hr, _, _ = w2.split()

        u_out.assign(self.au + self.dt*ur)
        h_out.assign(self.ah + self.dt*hr)


class SSPRK2V(NonlinearExponentialIntegrator):
    """
    u* = v* = u^n + dtN(u^n)
    u^{n+1} = exp(dtL)u^n + dt/2(exp(dtL)N(u^n) + N(exp(dtL)u*))
    """

    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half):
        super(SSPRK2V, self).__init__(setup, dt, direct_solve, h, M, reduce_to_half)
        self.ustar = Function(setup.spaces["u"])
        self.hstar = Function(setup.spaces["h"])
        self.u1 = Function(setup.spaces["u"])
        self.h1 = Function(setup.spaces["h"])

    def apply(self, u_in, h_in, u_out, h_out):

        dt = self.dt
        # calculate N(U^n) and U*
        self.u0.assign(u_in)
        self.h0.assign(h_in)
        self.nonlinear_solver.solve()
        Nu, Nh = self.Nw.split()        
        self.ustar = u_in + dt*Nu
        self.hstar = h_in + dt*Nh

        # calculate exp(dtL)U^n
        super(SSPRK2V, self).apply(u_in, h_in, u_out, h_out)

        # calculate exp(dtL)N(u^n)
        super(SSPRK2V, self).apply(Nu, Nh, self.u1, self.h1)
        u_out += 0.5*dt*self.u1
        h_out += 0.5*dt*self.h1

        # calculate N(exp(dtL)u*)
        super(SSPRK2V, self).apply(self.ustar, self.hstar, self.u1, self.h1)
        self.u0.assign(self.u1)
        self.h0.assign(self.h1)
        self.nonlinear_solver.solve()
        Nexpustar, Nexphstar = self.Nw.split()
        u_out += 0.5*dt*Nexpustar
        h_out += 0.5*dt*Nexphstar

class ETDRK4V(NonlinearExponentialIntegrator):
    """
    u1 = exp(0.5dtL)*(u^n + 0.5dtN(u^n))
    u2 = exp(0.5dtL)u^n + 0.5dtN(exp(0.5dtL)u1)
    u3 = exp(dtL)u^n + dt
    u^{n+1} = exp(dtL)u^n + dt/6(exp(dtL)N(u^n) 
              + 2exp(0.5dtL)(N(u1) + N(u2))
              + N(u3))
    """

    def __init__(self, setup, dt, direct_solve, h, M, reduce_to_half):
        super(ETDRK4V, self).__init__(setup, dt, direct_solve, h, M, reduce_to_half)
        self.u1 = Function(setup.spaces["u"])
        self.h1 = Function(setup.spaces["h"])
        self.u2 = Function(setup.spaces["u"])
        self.h2 = Function(setup.spaces["h"])
        self.u3 = Function(setup.spaces["u"])
        self.h3 = Function(setup.spaces["h"])

    def apply(self, u_in, h_in, u_out, h_out):

        dt = self.dt
        # calculate N(U^n)
        self.u0.assign(u_in)
        self.h0.assign(h_in)
        self.nonlinear_solver.solve()
        Nu, Nh = self.Nw.split()        

        # calculate exp(dtL)U^n
        super(ETDRK4V, self).apply(u_in, h_in, u_out, h_out)

        # calculate exp(0.5dtL)U^n
        super(ETDRK4V, self).apply(u_in, h_in, u_tmp, h_tmp, dt=0.5*dt)
        self.u1.assign(u_tmp)
        self.h1.assign(h_tmp)
        self.u2.assign(u_tmp)
        self.h2.assign(h_tmp)

        # calculate exp(dtL)N(u^n) and U1
        super(ETDRK4V, self).apply(Nu, Nh, expNu, expNh)
        self.u1 += 0.5*dt*expNu
        self.h1 += 0.5*dt*expNh

        # calculate N(U1) and U2
        self.u0.assign(self.u1)
        self.h0.assign(self.h1)
        self.nonlinear_solver.solve()
        Nu1, Nh1 = self.Nw.split()
        self.u2 += 0.5*dt*Nu1
        self.h2 += 0.5*dt*Nh1

        # calculate N(U2), exp(0.5*dt*L)N(u^2) and U3
        self.u3.assign(u_out)
        self.h3.assign(h_out)
        self.u0.assign(self.u2)
        self.h0.assign(self.h2)
        self.nonlinear_solver.solve()
        Nu2, Nh2 = self.Nw.split()
        super(ETDRK4V, self).apply(Nu2, Nh2, u_tmp, h_tmp, dt=0.5*dt)
        self.u3 += dt*u_tmp
        self.h3 += dt*h_tmp

        # calculate N(U3)
        self.u0.assign(self.u3)
        self.h0.assign(self.h3)
        self.nonlinear_solver.solve()
        Nu3, Nh3 = self.Nw.split()

        u_out += (dt/6.)*(expNu + 2*(expNu1 + expNu2) + Nu3)
        h_out += (dt/6.)*(expNh + 2*(expNh1 + expNh2) + Nh3)

class CoarsePropagator(NonlinearExponentialIntegrator):

    def __init__(self, setup, dt, direct_solve, h, rexiM, reduce_to_half, T, M, nonlinear=True):
        super(CoarsePropagator, self).__init__(setup, dt, direct_solve, h, rexiM, reduce_to_half, nonlinear)
        self.sn = np.arange(0.5*T/M, T, T/M)
        self.T = T
        self.u1 = Function(setup.spaces["u"])
        self.h1 = Function(setup.spaces["h"])
        self.u_tmp = Function(setup.spaces["u"])
        self.h_tmp = Function(setup.spaces["h"])

    def rho(self, t, C):
        rho0 = 0.*t
        w = np.where ((t < 1) & (t > 0))
        t1 = t[w]
        rho0[w] = C*np.exp(-1.0/(t1*(1.0-t1)))
        return rho0

    def apply(self, u0, h0, u_out, h_out):

        rho_sn = self.rho(self.sn/self.T, 1.0)
        rho_sn /= sum(rho_sn)
        u_out.assign(u0)
        h_out.assign(h0)
        self.u1.assign(u0)
        self.h1.assign(h0)

        # stage 1 of midpoint method
        for i, s in enumerate(self.sn):
            super(CoarsePropagator, self).apply(u0, h0, self.u_tmp, self.h_tmp, s)
            self.u0.assign(self.u_tmp)
            self.h0.assign(self.h_tmp)
            self.nonlinear_solver.solve()
            Nu, Nh = self.Nw.split()
            super(CoarsePropagator, self).apply(Nu, Nh, self.u_tmp, self.h_tmp, -s)
            self.u1 += 0.5*self.dt*Constant(rho_sn[i])*self.u_tmp
            self.h1 += 0.5*self.dt*Constant(rho_sn[i])*self.h_tmp

        # stage 2 of midpoint method
        for i, s in enumerate(self.sn):
            super(CoarsePropagator, self).apply(self.u1, self.h1, self.u_tmp, self.h_tmp, s)
            self.u0.assign(self.u_tmp)
            self.h0.assign(self.h_tmp)
            self.nonlinear_solver.solve()
            Nu, Nh = self.Nw.split()
            super(CoarsePropagator, self).apply(Nu, Nh, self.u_tmp, self.h_tmp, -s)
            u_out += self.dt*Constant(rho_sn[i])*self.u_tmp
            h_out += self.dt*Constant(rho_sn[i])*self.h_tmp
        super(CoarsePropagator, self).apply(u_out, h_out, u_out, h_out)
