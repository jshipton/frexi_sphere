from firedrake import *
from os import path
from rexi_coefficients import REXI
from sw_setup import SetupShallowWater

class RexiTimestep(object):

    def __init__(self, mesh, family, degree, problem_name, t, h, M, nonlinear=True, outward_normals=None, dirname='results'):

        self.dirname = dirname
        self.problem_name = problem_name
        self.dt = t
        self.outward_normals = outward_normals
        self.n = FacetNormal(mesh)

        self.alpha, self.beta_re, _ = REXI(h, M, reduce_to_half=False)
        self.nonlinear = nonlinear
        if nonlinear:
            self.alpha1, self.beta1_re, _ = REXI(h, M, n=1, reduce_to_half=False)
        if problem_name == "w5":
            bexpr = Expression("2000 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/R0,x[0]/R0)+1.0*pi/2.0,2)+pow(asin(x[2]/R0)-pi/6.0,2)))/(pi/9.0))", R0=6371220.)
            b = Function(V2).interpolate(bexpr)


        self.setup = SetupShallowWater(mesh, family, degree, problem_name)
        V1 = self.setup.spaces['u']
        V2 = self.setup.spaces['h']
        self.uout = Function(V1,name="u")
        self.hout = Function(V2,name="h")
        filename = path.join(dirname, 'rexi_'+problem_name+'_t'+str(t)+'_h'+str(h)+'_M'+str(M)+'.pvd')
        self.outfile = File(filename)


    def run(self, u0, h0, direct_solve=False):

        f = self.setup.params.f
        g = Constant(self.setup.params.g)
        H = Constant(self.setup.params.H)
        dt = Constant(self.dt)
        n = self.n
        Upwind = 0.5*(sign(dot(u0, n))+1)
        if self.outward_normals is not None:
            perp = lambda u: cross(self.outward_normals, u)
            perp_u_upwind = lambda q: Upwind('+')*cross(self.outward_normals('+'),q('+')) + Upwind('-')*cross(self.outward_normals('-'),q('-'))
        else:
            perp = lambda u: as_vector([-u[1], u[0]])
            perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))

        ai = Constant(1.0)
        bi = Constant(100.0)
        ar = Constant(1.0)
        br = Constant(100.0)

        V1 = self.setup.spaces['u']
        V2 = self.setup.spaces['h']
        W = MixedFunctionSpace((V1,V2,V1,V2))

        u1r, h1r, u1i, h1i = TrialFunctions(W)
        wr, phr, wi, phi = TestFunctions(W)

        a = (
            inner(wr,u1r)*ar - dt*f*inner(wr,perp(u1r)) + dt*g*div(wr)*h1r 
            - ai*inner(wr,u1i)
            + phr*(ar*h1r - dt*H*div(u1r) - ai*h1i)
            + inner(wi,u1i)*ar - dt*f*inner(wi,perp(u1i)) + dt*g*div(wi)*h1i 
            + ai*inner(wi,u1r)
            + phi*(ar*h1i - dt*H*div(u1i) + ai*h1r)
        )*dx

        L = (
            br*inner(wr,u0)*dx
            + br*phr*h0*dx 
            + bi*inner(wi,u0)*dx
            + bi*phi*h0*dx 
        )

        w = Function(W)
        myprob = LinearVariationalProblem(a, L, w, constant_jacobian=False)

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

        rexi_solver = LinearVariationalSolver(myprob,
                                              solver_parameters=solver_parameters)

        w_sum = Function(W)
        N = len(self.alpha)
        for i in range(N):
            ai.assign(self.alpha[i].imag)
            ar.assign(self.alpha[i].real)
            bi.assign(self.beta_re[i].imag)
            br.assign(self.beta_re[i].real)

            rexi_solver.solve()
            _,hr,_,_ = w.split()
            print i, ai.dat.data[0], ar.dat.data[0], bi.dat.data[0], br.dat.data[0], hr.dat.data.min(), hr.dat.data.max() 
            w_sum += w

        u1rl_,h1rl_,u1il_,h1il_ = w_sum.split()

        if self.nonlinear:
            gradperp = lambda u: perp(grad(u))

            u_adv_term =(
                -inner(gradperp(inner(wr, perp(u0))), u0)*dx
                - inner(jump(inner(wr, perp(u0)), n),
                        perp_u_upwind(u0))*dS
                -div(wr)*(h0 + 0.5*inner(u0, u0))*dx
            )
            h_cont_term = (
                (-dot(grad(phr), u0)*h0*dx +
                 jump(u0*phr, n)*avg(h0)*dS)
            )

            aNu = inner(wr, u1r)*dx + inner(phr, h1r)*dx
            LNu = u_adv_term + h_cont_term
            Nu = Function(W)
            solve(aNu == LNu, Nu)
            Nuu_, Nuh_, _, _ = Nu.split()
            u0.assign(Nuu_)
            h0.assign(Nuh_)
            w1_sum = Function(W)
            for i in range(N):
                ai.assign(self.alpha1[i].imag)
                ar.assign(self.alpha1[i].real)
                bi.assign(self.beta1_re[i].imag)
                br.assign(self.beta1_re[i].real)

                rexi_solver.solve()
                _,hr,_,_ = w.split()
                print i, ai.dat.data[0], ar.dat.data[0], bi.dat.data[0], br.dat.data[0], hr.dat.data.min(), hr.dat.data.max() 
                w1_sum += w
            u1r_,h1r_,u1i_,h1i_ = w1_sum.split()
            self.uout.assign(u1rl_ + dt*u1r_)
            self.hout.assign(h1rl_ + dt*h1r_)
        else:
            self.uout.assign(u1rl_)
            self.hout.assign(h1rl_)

        self.outfile.write(self.uout, self.hout)

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
