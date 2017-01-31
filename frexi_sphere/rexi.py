from firedrake import *
from rexi_coefficient_python import REXI
from sw_setup import SetupShallowWater

class RexiTimestep(object):

    def __init__(self, mesh, family, degree, problem_name, t):

        self.problem_name = problem_name
        self.dt = Constant(t)

        self.setup = SetupShallowWater(mesh, family, degree, problem_name)

    def run(self, h, M, direct_solve=False):

        filename = 'rexi_'+self.problem_name+'_t'+str(t)+'_h'+str(h)+'_M'+str(M)+'.pvd'
        f = Constant(self.setup.params.f)
        g = Constant(self.setup.params.g)
        H = Constant(self.setup.params.H)
        dt = self.dt

        rexi = REXI(h, M, i_reduce_to_half=False)

        ai = Constant(1.0)
        bi = Constant(100.0)
        ar = Constant(1.0)
        br = Constant(100.0)

        V1 = self.setup.spaces['u']
        V2 = self.setup.spaces['h']
        W = MixedFunctionSpace((V1,V2,V1,V2))

        u0 = Function(V1).assign(self.setup.u0)
        h0 = Function(V2).assign(self.setup.h0)

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
        N = len(rexi.alpha)
        for i in range(N):
            ai.assign(rexi.alpha[i].imag)
            ar.assign(rexi.alpha[i].real)
            bi.assign(rexi.beta_re[i].imag)
            br.assign(rexi.beta_re[i].real)

            rexi_solver.solve()
            _,hr,_,_ = w.split()
            print i, ai.dat.data[0], ar.dat.data[0], bi.dat.data[0], br.dat.data[0], hr.dat.data.min(), hr.dat.data.max() 
            w_sum += w

        u1r_,h1r_,u1i_,h1i_ = w_sum.split()

        self.u1r = Function(V1,name="u1r").assign(u1r_)
        u1i = Function(V1,name="u1i").assign(u1i_)
        self.h1r = Function(V2,name="h1r").assign(h1r_)
        h1i = Function(V2,name="h1i").assign(h1i_)

        File(filename).write(self.u1r, self.h1r, u1i, h1i)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description="setup shallow water solver")
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--sphere", nargs=2, type=float)
    geometry.add_argument("--square", nargs=2, type=float)
    parser.add_argument("problem_name")
    parser.add_argument("t", type=float)
    parser.add_argument("h", type=float)
    parser.add_argument("M", type=int)
    parser.add_argument("--family", default="BDM")
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--direct_solve", action="store_true")

    args = parser.parse_args()
    print args
    if args.square is not None:
        L = args.square[0]
        n = int(args.square[1])
        print "setting up square mesh of length %s with n %s"%(L, n)
        mesh = PeriodicSquareMesh(n, n, args.square[0])
        perp = lambda u: as_vector([-u[1], u[0]])
    elif args.sphere is not None:
        R = args.sphere[1]
        ref = int(args.sphere[0])
        print "setting up sphere mesh with radius %s and refinement level %s"%(R, ref)
        mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref, degree=1)
        global_normal = Expression(("x[0]", "x[1]", "x[2]"))
        mesh.init_cell_orientations(global_normal)
        outward_normals = CellNormal(mesh)
        perp = lambda u: cross(outward_normals, u)
    else:
        print "Geometry not recognised"

    r = RexiTimestep(mesh, args.family, args.degree, args.problem_name, args.t)
    r.run(args.h, args.M, args.direct_solve)