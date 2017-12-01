from firedrake import *

class Rexi(object):

    def __init__(self, setup, direct_solve, rexi_coefficients):

        alpha, beta_re = rexi_coefficients
        self.alpha = alpha
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
                        
            hybridisation_parameters = {'ksp_type': 'preonly',
                                        'pc_type': 'python',
                                        'pc_python_type': 'firedrake.HybridizationPC',
                                        'hybridization': {'ksp_type': 'preonly',
                                                          'pc_type': 'lu',
                                                          'hdiv_residual_ksp_type': 'preonly',
                                                          'hdiv_residual_pc_type': 'lu', 
                                                          'hdiv_projection_ksp_type': 'preonly',
                                                          'hdiv_projection_pc_type': 'lu'}}

            lu_parameters = {'ksp_type':'preonly',
                             'pc_type':'lu'}
            
            solver_parameters = {"ksp_type": "gmres",
                                 'mat_type': 'matfree',
                                 "ksp_converged_reason": True,
                                 "pc_type": "fieldsplit",
                                 "pc_fieldsplit_type": "multiplicative",
                                 "pc_fieldsplit_off_diag_use_amat": True,
                                 "pc_fieldsplit_0_fields": "0,1",
                                 "pc_fieldsplit_1_fields": "2,3",
                                 "fieldsplit_0": hybridisation_parameters,
                                 "fieldsplit_1": hybridisation_parameters}
            # For reusing solver with different A, but same aP.
            #solver_parameters["ksp_reuse_preconditioner"] = True

        self.w_sum = Function(W)
        self.w = Function(W)

        def L_op(u, h, v, q):
            lform = -dt*f*inner(v,perp(u))*dx
            lform += dt*g*div(v)*h*dx
            lform += -dt*H*q*div(u)*dx
            return lform

        def inner_m(u, h, v, q):
            return inner(u,v)*dx + h*q*dx

        ar0 = Constant(alpha[0].real)
        ai0 = Constant(alpha[0].imag)
        
        for i in range(len(alpha)):
            ai = Constant(alpha[i].imag)
            bi = Constant(beta_re[i].imag)
            ar = Constant(alpha[i].real)
            br = Constant(beta_re[i].real)

            # (1           -sgn(ai))*(ar + L    -ai   )
            # (sgn(ai)            1) (ai        ar + L)

            # (1,1) block
            a = (ar - abs(ai))*inner_m(u1r, h1r, wr, phr)
            a += L_op(u1r, h1r, wr, phr)
            # (1,2) block
            a += (-ai - sign(ai)*ar)*inner_m(u1i, h1i, wr, phr)
            a += -sign(ai)*L_op(u1i, h1i, wr, phr)
            # (2,1) block
            a += (ai + sign(ai)*ar)*inner_m(u1r, h1r, wi, phi)
            a += +sign(ai)*L_op(u1r, h1r, wi, phi)
            # (2,2) block
            a += (ar - abs(ai))*inner_m(u1i, h1i, wi, phi)
            a += L_op(u1i, h1i, wi, phi)

            # (1           -sgn(ai))*(br*inner)
            # (sgn(ai)            1) (bi*inner)
            
            L = (
                (br - sign(ai)*bi)*(inner(wr,self.u0)*dx
                                   + phr*self.h0*dx)
                +(+sign(ai)*br + bi)*(inner(wi,self.u0)*dx
                                      + phi*self.h0*dx)
            )

            # (1,1) block
            aP = (ar - abs(ai))*inner_m(u1r, h1r, wr, phr)
            aP += L_op(u1r, h1r, wr, phr)
            # (2,2) block
            aP += (ar - abs(ai))*inner_m(u1i, h1i, wi, phi)
            aP += L_op(u1i, h1i, wi, phi)
            
            myprob = LinearVariationalProblem(a, L, self.w, aP=aP)

            self.rexi_solver.append(LinearVariationalSolver(
                myprob, solver_parameters=solver_parameters))

    def solve(self, u0, h0, dt):
        self.u0.assign(u0)
        self.h0.assign(h0)
        self.dt.assign(dt)

        self.w_sum.assign(0.)

        for i in range(len(self.rexi_solver)):
            print(self.alpha[i])
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
