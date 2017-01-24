from firedrake import *
from rexi_coefficient_python import REXI

Mesh = IcosahedralSphereMesh(radius=1.0, refinement_level=5, degree=1)

global_normal = Expression(("x[0]", "x[1]", "x[2]"))
Mesh.init_cell_orientations(global_normal)

V1 = FunctionSpace(Mesh,"BDM",1)
V2 = FunctionSpace(Mesh,"DG",0)

f = Constant(1.0)
g = Constant(9.8)
H = Constant(1.0)

h = 0.2
M = 64
rexi = REXI(h, M)

ai = Constant(1.0)
bi = Constant(100.0)
ar = Constant(1.0)
br = Constant(100.0)
dt = Constant(1.)

W = MixedFunctionSpace((V1,V2,V1,V2))

u0 = Function(V1)
h0 = Function(V2).interpolate(Expression("1.0"))

u1r, h1r, u1i, h1i = TrialFunctions(W)
wr, phr, wi, phi = TestFunctions(W)

outward_normals = CellNormal(Mesh)
perp = lambda u: cross(outward_normals, u)

a = (
    inner(wr,u1r)*ar - dt*f*inner(wr,perp(u1r)) + dt*g*div(wr)*h1r 
    - ai*inner(wr,u1i)
    + phr*(ai*h1r - dt*H*div(u1r) - ar*h1i)
    + inner(wi,u1i)*ar - dt*f*inner(wi,perp(u1i)) + dt*g*div(wi)*h1i 
    + ai*inner(wi,u1r)
    + phi*(ai*h1i - dt*H*div(u1i) + ar*h1r)
)*dx

L = (
    br*inner(wr,u0)*dx
    + br*phr*h0*dx 
    + bi*inner(wi,u0)*dx
    + bi*phi*h0*dx 
    )

w = Function(W)
myprob = LinearVariationalProblem(a,L,w)

block_lu_solver_parameters = {"ksp_type": "gmres",
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
                                      solver_parameters=block_lu_solver_parameters)

w_sum = Function(W)

for i in range(len(rexi.alpha)):
    ai.assign(rexi.alpha[i].imag)
    ar.assign(rexi.alpha[i].real)
    bi.assign(rexi.beta_re[i].imag)
    br.assign(rexi.beta_re[i].real)

    rexi_solver.solve()
    w_sum += w

u1r_,h1r_,u1i_,h1i_ = w_sum.split()

u1r = Function(V1,name="u1r").assign(u1r_)
u1i = Function(V1,name="u1i").assign(u1i_)
h1r = Function(V2,name="h1r").assign(h1r_)
h1i = Function(V2,name="h1i").assign(h1i_)

File('rexi.pvd').write(u1r,h1r,u1i,h1i)
