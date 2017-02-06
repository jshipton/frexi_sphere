from firedrake import *
from frexi_sphere.rexi import RexiTimestep

R = 6371220.
t = 0.
tmax = 90000.
dt = 1500.
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=3)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)
outward_normals = CellNormal(mesh)

h = 0.2
M = 64
timestep = RexiTimestep(mesh, "BDM", 1, "w2", dt, h, M, outward_normals=outward_normals)
V1 = timestep.setup.spaces['u']
V2 = timestep.setup.spaces['h']
u_init = Function(V1,name="u_init").assign(timestep.setup.u0)
h_init = Function(V2,name="h_init").assign(timestep.setup.h0)
u_err = Function(V1,name="u_err")
h_err = Function(V2,name="h_err")
u0 = Function(V1,name="u").assign(timestep.setup.u0)
h0 = Function(V2,name="h").assign(timestep.setup.h0)

while t - 0.5*dt < tmax:
    
    t += dt
    timestep.run(u0, h0, True)
    u0.assign(timestep.uout)
    h0.assign(timestep.hout)
    u_err.assign(timestep.uout - u_init)
    h_err.assign(timestep.hout - h_init)
