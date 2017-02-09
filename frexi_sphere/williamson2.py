from firedrake import *
from frexi_sphere.exponential_integrators import ETD1, ETD2RK
from frexi_sphere.sw_setup import SetupShallowWater
from os import path

# set up mesh and initial conditions for Williamson 2 testcase
R = 6371220.
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=3)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)
setup = SetupShallowWater(mesh, family="BDM", degree=1, problem_name="w2")

# get spaces and initialise
V1 = setup.spaces['u']
V2 = setup.spaces['h']
u0 = Function(V1,name="u").assign(setup.u0)
h0 = Function(V2,name="h").assign(setup.h0)
u_out = Function(V1)
h_out = Function(V2)

# save initial conditions for computing errors and set up error fields
u_init = Function(V1,name="u_init").assign(setup.u0)
h_init = Function(V2,name="h_init").assign(setup.h0)
u_err = Function(V1,name="u_err")
h_err = Function(V2,name="h_err")

t = 0.
tmax = 5*24.*60.*60.
dt = 3000.

h = 0.2
M = 64
timestepper = ETD2RK(setup, dt, True, h, M, False)

filename = path.join('results', 'rexi_w2_dt'+str(dt)+'_h'+str(h)+'_M'+str(M)+'.pvd')
outfile = File(filename)
outfile.write(u0, h0, u_err, h_err)

while t - 0.5*dt < tmax:
    
    t += dt
    timestepper.apply(u0, h0, u_out, h_out)
    u0.assign(u_out)
    h0.assign(h_out)
    u_err.assign(u_out - u_init)
    h_err.assign(h_out - h_init)

    outfile.write(u0, h0, u_err, h_err)


