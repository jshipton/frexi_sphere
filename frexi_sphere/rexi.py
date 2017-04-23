from firedrake import *

class Rexi(object):

    def __init__(self, setup, direct_solve):

        V1 = setup.spaces['u']
        V2 = setup.spaces['h']
        self.u0 = Function(V1, name="u")
        self.h0 = Function(V2, name="h")

        f = setup.params.f
        g = Constant(setup.params.g)
        H = Constant(setup.params.H)
        self.dt = Constant(1.)
        dt = self.dt

        if setup.outward_normals is not None:
            perp = lambda u: cross(setup.outward_normals, u)
        else:
            perp = lambda u: as_vector([-u[1], u[0]])

        self.ai = Constant(1.0)
        self.bi = Constant(100.0)
        self.ar = Constant(1.0)
        self.br = Constant(100.0)

        W = MixedFunctionSpace((V1,V2,V1,V2))

        u1r, h1r, u1i, h1i = TrialFunctions(W)
        wr, phr, wi, phi = TestFunctions(W)

        a = (
            inner(wr,u1r)*self.ar - dt*f*inner(wr,perp(u1r)) + dt*g*div(wr)*h1r 
            - self.ai*inner(wr,u1i)
            + phr*(self.ar*h1r - dt*H*div(u1r) - self.ai*h1i)
            + inner(wi,u1i)*self.ar - dt*f*inner(wi,perp(u1i)) + dt*g*div(wi)*h1i 
            + self.ai*inner(wi,u1r)
            + phi*(self.ar*h1i - dt*H*div(u1i) + self.ai*h1r)
        )*dx

        L = (
            self.br*inner(wr,self.u0)*dx
            + self.br*phr*self.h0*dx 
            + self.bi*inner(wi,self.u0)*dx
            + self.bi*phi*self.h0*dx 
        )

        self.w_sum = Function(W)
        self.w = Function(W)
        myprob = LinearVariationalProblem(a, L, self.w, constant_jacobian=False)

        if direct_solve:
            solver_parameters = {'ksp_type':'preonly',
                                 'mat_type': 'aij',
                                 'pc_type':'lu',
                                 'pc_factor_mat_solver_package': 'mumps'}
        else:
            solver_parameters = {"ksp_type": "gmres",
                                 "ksp_monitor": True,
                                 "pc_type": "fieldsplit",
                                 "mat_type": "aij",
                                 "pc_fieldsplit_type": "multiplicative",
                                 "pc_fieldsplit_0_fields": "0,1",
                                 "pc_fieldsplit_1_fields": "2,3",
                                 "fieldsplit_0_ksp_type": "preonly",
                                 "fieldsplit_1_ksp_type": "preonly",
                                 "fieldsplit_0_pc_type": "lu",
                                 "fieldsplit_1_pc_type": "lu"}

        self.rexi_solver = LinearVariationalSolver(
            myprob, solver_parameters=solver_parameters)

    def solve(self, u0, h0, dt, rexi_coefficients):
        self.u0.assign(u0)
        self.h0.assign(h0)
        alpha, beta_re = rexi_coefficients
        self.dt.assign(dt)

        N = len(alpha)
        self.w_sum.assign(0.)
        for i in range(N):
            self.ai.assign(alpha[i].imag)
            self.ar.assign(alpha[i].real)
            self.bi.assign(beta_re[i].imag)
            self.br.assign(beta_re[i].real)

            self.rexi_solver.solve()

            self.w_sum += self.w

        return self.w_sum


if __name__=="__main__":
    from input_parsing import RexiArgparser
    rargs = RexiArgparser()
    mesh = rargs.mesh
    try:
        outward_normals = rargs.outward_normals
    except AttributeError:
        outward_normals = None
    args = rargs.args
    r = RexiTimestep(mesh, args.family, args.degree, args.problem_name, args.t, outward_normals=outward_normals)
    r.run(args.h, args.M, args.direct_solve)
