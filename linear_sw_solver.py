from firedrake import *
from sw_setup import SetupShallowWater

import argparse

parser = argparse.ArgumentParser(description="setup shallow water solver")
geometry = parser.add_mutually_exclusive_group()
geometry.add_argument("--sphere", nargs=2, type=float)
geometry.add_argument("--square", nargs=2, type=float)
parser.add_argument("problem_name")
parser.add_argument("dt", type=float)
parser.add_argument("tmax", type=float)
parser.add_argument("--family", default="BDM")
parser.add_argument("--degree", type=int, default=0)

args = parser.parse_args()

if args.square is not None:
    L = args.square[0]
    n = int(args.square[1])
    print "setting up square mesh of length %s with n %s"%(L, n)
    mesh = PeriodicSquareMesh(n, n, args.square[0])
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

setup = SetupShallowWater(mesh, args.family, args.degree, args.problem_name)

V1 = setup.V1
V2 = setup.V2
W = MixedFunctionSpace((V1, V2))
f = setup.params.f
g = setup.params.g
H = setup.params.H

dt = args.dt
tmax = args.tmax

uh0 = Function(W)
u0, h0 = uh0.split()
u0.assign(setup.u0)
h0.assign(setup.h0)

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

filename = args.problem_name + "_out.pvd"
outfile = File(filename)
u1.rename('velocity')
h1.rename('height')
outfile.write(u1, h1)
while t < tmax - 0.5*dt:

    print "t = ", t, "energy = ", assemble(0.5*(inner(u1, u1) + g*H*h1*h1)*dx)
    solver.solve()
    u1, h1 = uh1.split()

    outfile.write(u1, h1)
    t += dt

print "t = ", t, "energy = ", assemble(0.5*(inner(u1, u1) + g*H*h1*h1)*dx)

