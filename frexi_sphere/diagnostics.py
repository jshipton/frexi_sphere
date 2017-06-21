from firedrake import assemble, inner, dx, sqrt, op2, dot, \
    FunctionSpace, Function, TestFunction
import numpy as np


class Diagnostics(object):

    @staticmethod
    def min(f_in):
        fmin = op2.Global(1, [np.finfo(float).max], dtype=float)
        if len(f_in.ufl_shape) > 0:
            mesh = f_in.function_space().mesh()
            V = FunctionSpace(mesh, "DG", 1)
            f = Function(V).project(sqrt(inner(f_in, f_in)))
        else:
            f = f_in
        op2.par_loop(op2.Kernel("""void minify(double *a, double *b)
        {
        a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
        }""", "minify"),
                     f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
        return fmin.data[0]

    @staticmethod
    def max(f_in):
        fmax = op2.Global(1, [np.finfo(float).min], dtype=float)
        if len(f_in.ufl_shape) > 0:
            mesh = f_in.function_space().mesh()
            V = FunctionSpace(mesh, "DG", 1)
            f = Function(V).project(sqrt(inner(f_in, f_in)))
        else:
            f = f_in
        op2.par_loop(op2.Kernel("""void maxify(double *a, double *b)
        {
        a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
        }""", "maxify"),
                     f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
        return fmax.data[0]

    @staticmethod
    def l2(f):
        return sqrt(assemble(dot(f, f)*dx))

    @staticmethod
    def energy(h, u, g):
        return assemble(0.5*h*(inner(u, u) + g*h)*dx)

    def max_courant_number(self, u, dt):
        if not hasattr(self, "_area"):
            V = FunctionSpace(u.function_space().mesh(), "DG", 0)
            expr = TestFunction(V)*dx
            self._area = assemble(expr)
            self.Courant = Function(V)
        self.Courant.project(sqrt(inner(u, u))/sqrt(self._area)*dt)
        return self.Courant.dat.data.max()
