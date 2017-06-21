from firedrake import *
from os import path


class ImplicitMidpointLinearSWSolver(object):

    def __init__(self, setup, dt, outward_normals=None, dirname='results'):

        self.dt = dt
        self.filename = path.join(dirname, 'imsolve_dt'+str(dt)+'.pvd')
        self.outward_normals = outward_normals
        self.setup = setup

    def run(self, tmax):

        f = self.setup.params.f
        g = Constant(self.setup.params.g)
        H = Constant(self.setup.params.H)
        dt = self.dt
        if self.outward_normals is not None:
            perp = lambda u: cross(self.outward_normals, u)
        else:
            perp = lambda u: as_vector([-u[1], u[0]])

        V1 = self.setup.spaces['u']
        V2 = self.setup.spaces['h']
        W = MixedFunctionSpace((V1, V2))

        uh0 = Function(W)
        u0, h0 = uh0.split()
        u0.assign(self.setup.u0)
        h0.assign(self.setup.h0)

        w, phi = TestFunctions(W)
        u, h = TrialFunctions(W)
        uh1 = Function(W)
        u1, h1 = uh1.split()
        ustar = 0.5*(u + u1)
        hstar = 0.5*(h + h1)
        eqn = (
            (inner(w, u - u1) + dt*(f*inner(w, perp(ustar)) - g*div(w)*hstar) +
             phi*(h - h1) + dt*H*inner(phi, div(ustar)))*dx
        )

        a = lhs(eqn)
        L = rhs(eqn)
        prob = LinearVariationalProblem(a, L, uh1)
        solver = LinearVariationalSolver(prob)

        t = 0.

        u1.assign(u0)
        h1.assign(h0)

        outfile = File(self.filename)
        u1.rename('velocity')
        h1.rename('height')
        outfile.write(u1, h1)
        print t, tmax-0.5*dt
        while t < tmax - 0.5*dt:

            print "t = ", t, "energy = ", \
                assemble(0.5*(inner(u1, u1) + g*H*h1*h1)*dx)

            solver.solve()
            u1, h1 = uh1.split()
            outfile.write(u1, h1)

            t += dt

        print "t = ", t, "energy = ", \
            assemble(0.5*(inner(u1, u1) + g*H*h1*h1)*dx)

        outfile.write(u1, h1)
        self.u_end = u1
        self.h_end = h1

if __name__ == "__main__":
    from input_parsing import ImplicitMidpointArgparser
    imargs = ImplicitMidpointArgparser()
    mesh = imargs.mesh
    try:
        outward_normals = imargs.outward_normals
    except AttributeError:
        outward_normals = None
    args = imargs.args

    im = ImplicitMidpointLinearSWSolver(mesh, args.family, args.degree, args.problem_name, args.dt, outward_normals=outward_normals)
    im.run(args.tmax)
