from firedrake import *
import numpy


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
        # ai = aapctx['ai']
        aimax = aapctx['aimax']
        perp = aapctx['perp']

        self.y_fn = Function(W)
        self.x_fn = Function(W)

        M = MixedFunctionSpace((V1, V2))
        self.vR = Function(M)
        self.vI = Function(M)

        self.uR = Function(M)
        self.uI = Function(M)

        self.aimax = aimax

        u, h = TrialFunctions(M)
        v, q = TestFunctions(M)

        a = (
            inner(v, u)*(ar + abs(aimax)) -
            dt*f*inner(v, perp(u)) + dt*g*div(v)*h +
            q*((ar + abs(aimax))*h - dt*H*div(u))
        )*dx

        hybridisation_parameters = {
            'ksp_type': 'preonly',
            'ksp_monitor': True,
            'mat_type': 'matfree',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.HybridizationPC',
            'hybridization': {'ksp_type': 'preonly',
                              'ksp_monitor': True,
                              'pc_type': 'hypre',
                              'hdiv_residual_ksp_type': 'preonly',
                              'hdiv_residual_pc_type': 'sor',
                              'hdiv_projection_ksp_type': 'preonly',
                              'hdiv_projection_pc_type': 'sor'}}

        # lu_parameters = {
        #     'ksp_type': 'preonly',
        #     'pc_type': 'lu',
        #     'pc_factor_mat_solver_package': 'mumps'}

        self.pc_solver = LinearSolver(
            assemble(a, mat_type='matfree'),
            solver_parameters=hybridisation_parameters)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError('We do not provide a transpose')

    def update(self, pc):
        """Preconditioner is always the same."""
        pass

    def apply(self, pc, y, x):

        with self.y_fn.dat.vec as yvec:
            y.copy(yvec)

        ur_in, hr_in, ui_in, hi_in = self.y_fn.split()
        ur_RHS, hr_RHS = self.vR.split()
        ui_RHS, hi_RHS = self.vI.split()
        # apply the mixture transformation
        ur_RHS.assign(ur_in + sign(self.aimax)*ui_in)
        hr_RHS.assign(hr_in + sign(self.aimax)*hi_in)
        ui_RHS.assign(-sign(self.aimax)*ur_in + ui_in)
        hi_RHS.assign(-sign(self.aimax)*hr_in + hi_in)
        # apply the solvers
        self.pc_solver.solve(self.uR, self.vR)
        self.pc_solver.solve(self.uI, self.vI)
        # copy back to x
        ur, hr = self.uR.split()
        ui, hi = self.uI.split()
        ur_out, hr_out, ui_out, hi_out = self.x_fn.split()
        ur_out.assign(ur)
        hr_out.assign(hr)
        ui_out.assign(ui)
        hi_out.assign(hi)

        with self.x_fn.dat.vec_ro as xvec:
            xvec.copy(x)


class Rexi(object):

    def __init__(self, setup, direct_solve, rexi_coefficients, IPcoeff=None):

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

        W = MixedFunctionSpace((V1, V2, V1, V2))

        u1r, h1r, u1i, h1i = TrialFunctions(W)
        wr, phr, wi, phi = TestFunctions(W)

        self.rexi_solver = []

        if direct_solve:
            solver_parameters = {'ksp_type': 'preonly',
                                 'mat_type': 'aij',
                                 'pc_type': 'lu',
                                 'pc_factor_mat_solver_package': 'mumps'}
        else:
            solver_parameters = {"ksp_type": "bcgs",
                                 "ksp_converged_reason": True,
                                 "mat_type": "matfree",
                                 "pc_type": "python",
                                 "pc_python_type": "rexi.REXI_PC"}

        self.ai = Constant(alpha[0].imag)
        self.bi = Constant(beta_re[0].imag)
        self.ar = Constant(alpha[0].real)
        self.br = Constant(beta_re[0].real)
        ai = self.ai
        ar = self.ar
        bi = self.bi
        br = self.br

        self.alpha = alpha
        self.beta_re = beta_re

        self.w_sum = Function(W)
        self.w = Function(W)

        solver = 'new'
        if solver == 'new' and not(direct_solve):
            assert(IPcoeff is not None)

            aimax = numpy.array(alpha).imag.max()

            def L_op(u, h, v, q):
                lform = -dt*f*inner(v, perp(u))*dx
                lform += dt*g*div(v)*h*dx
                lform += -dt*H*q*div(u)*dx
                return lform

            def inner_m(u, h, v, q):
                return inner(u, v)*dx + h*q*dx

            ac = ar + abs(aimax)
            sigma = dt**2*H*g/ac/(1 + (dt*f/ac)**2)

            IPcoeff = Constant(IPcoeff)

            # (1            sgn(ai))*(ar + L    -ai   )
            # (-sgn(ai)           1) (ai        ar + L)

            # (1,1) block
            a = (ar + abs(ai))*inner_m(u1r, h1r, wr, phr)
            a += L_op(u1r, h1r, wr, phr)
            # (1,2) block
            a += (-ai + sign(ai)*ar)*inner_m(u1i, h1i, wr, phr)
            a += sign(ai)*L_op(u1i, h1i, wr, phr)
            # (2,1) block
            a += (ai - sign(ai)*ar)*inner_m(u1r, h1r, wi, phi)
            a += -sign(ai)*L_op(u1r, h1r, wi, phi)
            # (2,2) block
            a += (ar + abs(ai))*inner_m(u1i, h1i, wi, phi)
            a += L_op(u1i, h1i, wi, phi)

            L = (
                (br + sign(ai)*bi)*(inner(wr, self.u0)*dx
                                    + phr*self.h0*dx)
                + (-sign(ai)*br + bi)*(inner(wi, self.u0)*dx
                                       + phi*self.h0*dx)
            )

            # a u - dt * f * perp (u) - dt * g * grad h = 0
            # a h - dt * H * div (u) = R_h
            # assume f constant, then
            # a grad(phi) + dt*f*grad(psi) - dt*g*grad(h) = 0
            # a gradperp(psi) - dt*f*gradperp(phi) = 0
            # a grad(psi) = dt*f*grad(phi)
            # grad(phi) + (dt*f/a)**2*grad(phi) - dt*g/a*grad(h) = 0
            # grad(phi) = dt*g/a/(1 + (dt*f/a)**2)*grad(h)
            # a h - div(dt**2*H*g/a/(1+ (dt*f/a)**2)*grad(h)) = R_h

            aP = ac*inner(u1r, wr)*dx
            aP += -dt*f*inner(wr, perp(u1r))*dx
            aP += (ac*phr*h1r + inner(grad(phr), sigma*grad(h1r)))*dx
            aP += IPcoeff*sigma*jump(phr)*jump(h1r)*dS
            aP += ac*inner(u1i, wi)*dx
            aP += -dt*f*inner(wi, perp(u1i))*dx
            aP += (ac*phi*h1i + inner(grad(phi), sigma*grad(h1i)))*dx
            aP += IPcoeff*sigma*jump(phi)*jump(h1i)*dS

            myprob = LinearVariationalProblem(a, L, self.w, aP=aP,
                                              constant_jacobian=False)

            # two_block_parameters = {
            #     'ksp_type': 'gmres',
            #     'pc_type': 'fieldsplit',
            #     'mat_type': 'aij',
            #     'ksp_monitor': False,
            #     'ksp_converged_reason': True,
            #     'pc_fieldsplit_type': 'multiplicative',
            #     'pc_fieldsplit_0_fields': '0,1',
            #     'pc_fieldsplit_1_fields': '2,3',
            #     'fieldsplit_0_ksp_type': 'preonly',
            #     'fieldsplit_1_ksp_type': 'preonly',
            #     'fieldsplit_0_pc_type': 'lu',
            #     'fieldsplit_1_pc_type': 'lu',
            #     'fieldsplit_0_pc_factor_mat_solver_package': 'mumps',
            #     'fieldsplit_1_pc_factor_mat_solver_package': 'mumps'}

            four_block_parameters = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'ksp_monitor': True,
                'ksp_converged_reason': True,
                'pc_fieldsplit_type': 'multiplicative',
                'pc_fieldsplit_0_fields': '1',
                'pc_fieldsplit_1_fields': '3',
                'pc_fieldsplit_0_fields': '0',
                'pc_fieldsplit_1_fields': '2',
                'fieldsplit_0_ksp_type': 'preonly',
                'fieldsplit_1_ksp_type': 'preonly',
                'fieldsplit_2_ksp_type': 'preonly',
                'fieldsplit_3_ksp_type': 'preonly',
                'fieldsplit_0_pc_type': 'ilu',
                'fieldsplit_1_pc_type': 'ilu',
                'fieldsplit_2_pc_type': 'ilu',
                'fieldsplit_3_pc_type': 'ilu'}

            self.rexi_solver = LinearVariationalSolver(
                myprob, solver_parameters=four_block_parameters)

        else:
            a = (
                inner(wr, u1r)*ar - dt*f*inner(wr, perp(u1r))
                + dt*g*div(wr)*h1r
                - ai*inner(wr, u1i)
                + phr*(ar*h1r - dt*H*div(u1r) - ai*h1i)
                + inner(wi, u1i)*ar - dt*f*inner(wi, perp(u1i))
                + dt*g*div(wi)*h1i
                + ai*inner(wi, u1r)
                + phi*(ar*h1i - dt*H*div(u1i) + ai*h1r)
            )*dx

            L = (
                br*inner(wr, self.u0)*dx
                + br*phr*self.h0*dx
                + bi*inner(wi, self.u0)*dx
                + bi*phi*self.h0*dx
            )

            myprob = LinearVariationalProblem(a, L, self.w,
                                              constant_jacobian=False)

            if(direct_solve):
                self.rexi_solver = LinearVariationalSolver(
                    myprob, solver_parameters=solver_parameters)
            else:
                # Pack in context variables for the preconditioner
                appctx = {'W': W, 'V1': V1, 'V2': V2, 'dt': dt,
                          'H': H, 'g': g, 'f': f, 'ar': ar,
                          'ai': ai, 'aimax': aimax, 'perp': perp}
                self.rexi_solver = LinearVariationalSolver(
                    myprob, solver_parameters=solver_parameters,
                    appctx=appctx)

    def solve(self, u0, h0, dt):
        self.u0.assign(u0)
        self.h0.assign(h0)
        self.dt.assign(dt)

        self.w_sum.assign(0.)
        for i in range(len(self.alpha)):
            print "in the loop", i
            self.ar.assign(self.alpha[i].real)
            self.ai.assign(self.alpha[i].imag)
            self.br.assign(self.beta_re[i].real)
            self.bi.assign(self.beta_re[i].imag)
            # self.test_solver.solve()
            self.rexi_solver.solve()
            self.w_sum += self.w

        return self.w_sum


if __name__ == "__main__":
    from input_parsing import RexiArgparser
    rargs = RexiArgparser()
    mesh = rargs.mesh
    try:
        outward_normals = rargs.outward_normals
    except AttributeError:
        outward_normals = None
    args = rargs.args
    r = RexiTimestep(mesh, args.family, args.degree, args.problem_name,
                     args.t, outward_normals=outward_normals)
    r.run(args.h, args.M, args.direct_solve)
