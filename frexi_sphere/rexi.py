from firedrake import *

class REXI_PC(PCBase):
    def initialize(self, pc):
        _, P = pc.getOperators()
        context = P.getPythonContext()
        aapctx = context.appctx

        W = aapctx['W']
        V1 = aapctx['V1']
        V2 = aapctx['V2']
        dt = aapctx['dt']
        H = aapctx['H']
        g = aapctx['g']
        f = aapctx['f']
        ar = aapctx['ar']
        ai = aapctx['ai']
        perp = aapctx['perp']

        self.y_fn = Function(W)
        self.x_fn = Function(W)

        M = MixedFunctionSpace((V1, V2))
        self.vR = Function(M)
        self.vI = Function(M)

        self.uR = Function(M)
        self.uI = Function(M)
        
        self.ar = ar
        self.ai = ai

        u, h = TrialFunctions(M)
        v, q = TestFunctions(M)

        a = (
            inner(v,u)*(ar - abs(ai)) - dt*f*inner(v,perp(u)) + dt*g*div(v)*h
            + q*((ar - abs(ai))*h - dt*H*div(u))
            )*dx

        self.pc_solver = LinearSolver(assemble(a, mat_type='aij'),
                                      solver_parameters={
                                          'ksp_type':'preonly',
                                          'pc_type':'lu',
                                          'pc_factor_mat_solver_package': 
                                          'mumps'})

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError('We do not provide a transpose')

    def update(self, pc):
        print "Update"
        raise NotImplementedError('We do not provide an update')

    def apply(self, pc, y, x):

        with self.y_fn.dat.vec as yvec:
            yvec.array[:] = y.array[:]

        ur_in, hr_in, ui_in, hi_in = self.y_fn.split()
        ur_RHS, hr_RHS = self.vR.split()
        ui_RHS, hi_RHS = self.vI.split()
        #apply the mixture transformation
        ur_RHS.assign( ur_in - sign(self.ai)*ui_in )
        hr_RHS.assign( hr_in - sign(self.ai)*hi_in )
        ui_RHS.assign( sign(self.ai)*ur_in + ui_in )
        hi_RHS.assign( sign(self.ai)*hr_in + hi_in )
        #apply the solvers
        self.pc_solver.solve(self.uR, self.vR)
        self.pc_solver.solve(self.uI, self.vI)
        #copy back to x
        ur, hr = self.uR.split()
        ui, hi = self.uI.split()
        ur_out, hr_out, ui_out, hi_out = self.x_fn.split()
        ur_out.assign(ur)
        hr_out.assign(hr)
        ui_out.assign(ui)
        hi_out.assign(hi)

        with self.x_fn.dat.vec_ro as xvec:
            x.array[:] = xvec.array[:]

class Rexi(object):

    def __init__(self, setup, direct_solve, rexi_coefficients):

        alpha, beta_re = rexi_coefficients

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

        W = MixedFunctionSpace((V1,V2,V1,V2))

        u1r, h1r, u1i, h1i = TrialFunctions(W)
        wr, phr, wi, phi = TestFunctions(W)

        self.rexi_solver = []
        
        if direct_solve:
            solver_parameters = {'ksp_type':'preonly',
                                 'mat_type': 'aij',
                                 'pc_type':'lu',
                                 'pc_factor_mat_solver_package': 'mumps'}
        else:
            solver_parameters = {"ksp_type": "gmres",
                                 "ksp_converged_reason": True,
                                 "mat_type":"matfree",
                                 "pc_type": "python",
                                 "pc_python_type": "rexi.REXI_PC"}

        for i in range(len(alpha)):
            ai = Constant(alpha[i].imag)
            bi = Constant(beta_re[i].imag)
            ar = Constant(alpha[i].real)
            br = Constant(beta_re[i].real)
            
            a = (
                inner(wr,u1r)*ar - dt*f*inner(wr,perp(u1r)) + dt*g*div(wr)*h1r 
                - ai*inner(wr,u1i)
                + phr*(ar*h1r - dt*H*div(u1r) - ai*h1i)
                + inner(wi,u1i)*ar - dt*f*inner(wi,perp(u1i)) + dt*g*div(wi)*h1i 
                + ai*inner(wi,u1r)
                + phi*(ar*h1i - dt*H*div(u1i) + ai*h1r)
            )*dx
            
            L = (
                br*inner(wr,self.u0)*dx
                + br*phr*self.h0*dx 
                + bi*inner(wi,self.u0)*dx
                + bi*phi*self.h0*dx 
            )

            self.w_sum = Function(W)
            self.w = Function(W)
            myprob = LinearVariationalProblem(a, L, self.w)

            if(direct_solve):
                self.rexi_solver.append(LinearVariationalSolver(
                    myprob, solver_parameters=solver_parameters))
            else:
                #Pack in context variables for the preconditioner
                appctx = {'W':W,'V1':V1,'V2':V2,'dt':dt,
                          'H':H, 'g':g, 'f':f, 'ar':ar,
                          'ai':ai, 'perp':perp}
                self.rexi_solver.append(LinearVariationalSolver(
                    myprob, solver_parameters=solver_parameters,
                                        appctx=appctx))

    def solve(self, u0, h0, dt):
        self.u0.assign(u0)
        self.h0.assign(h0)
        self.dt.assign(dt)

        self.w_sum.assign(0.)
        for i in range(len(self.rexi_solver)):
            self.rexi_solver[i].solve()
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
