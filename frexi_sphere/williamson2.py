from firedrake import *
from frexi_sphere.exponential_integrators import ETD1, ETD2RK
from frexi_sphere.sw_setup import SetupShallowWater
import frexi_sphere.diagnostics
from os import path
import json

# set up mesh and initial conditions for Williamson 2 testcase
R = 6371220.
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=3)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)
degree = 1
setup = SetupShallowWater(mesh, family="BDM", degree=degree, problem_name="w2")

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

# setup parameters for timestepping and REXI
t = 0.
dt = 3000.
tmax = 5*24.*60.*60.
h = 0.2
M = 64

# make timestepper
timestepper = ETD2RK(setup, dt, True, h, M, False)

# output file and output fields
dirname = 'w2_deg%s_dt%s_h%s_M%s' % (degree, dt, h, M)
filename = path.join('results/'+dirname, 'output.pvd')
outfile = File(filename)
fields = [u0, h0, u_err, h_err]
outfile.write(*fields)

# setup diagnostics
field_dict = {field.name(): field for field in fields}
diagnostics = frexi_sphere.diagnostics.Diagnostics()
diagnostics_list = ['max', 'min', 'l2']
diagnostics_dict = {name: getattr(diagnostics, name) for name in diagnostics_list}
diagnostics_data = {}
for name, field in field_dict.iteritems():
    diagnostics_data[name] = {}
    for diagnostic in diagnostics_list:
        diagnostics_data[name][diagnostic] = []

diagnostics_data['time'] = []
diagnostics_data['energy'] = []

# write out initial diagnostics
for fname, field in field_dict.iteritems():
    for dname, diagnostic in diagnostics_dict.iteritems():
        diagnostics_data[fname][dname].append(diagnostic(field))

diagnostics_data['energy'].append(diagnostics.energy(h0, u0, setup.params.g))
diagnostics_data['time'].append(t)

# print time and energy to check things are going well    
print t, diagnostics_data['energy'][-1]

# timestepping loop
while t < tmax - 0.5*dt:
    
    timestepper.apply(u0, h0, u_out, h_out)
    u0.assign(u_out)
    h0.assign(h_out)
    u_err.assign(u_out - u_init)
    h_err.assign(h_out - h_init)

    outfile.write(*fields)
    t += dt

    for fname, field in field_dict.iteritems():
        for dname, diagnostic in diagnostics_dict.iteritems():
            diagnostics_data[fname][dname].append(diagnostic(field))
    diagnostics_data['energy'].append(diagnostics.energy(h0, u0, setup.params.g))
    diagnostics_data['time'].append(t)

    # print time and energy to check things are going well    
    print t, diagnostics_data['energy'][-1]

# dump diagnostics dictionary
with open(path.join("results/"+dirname, "diagnostics.json"), "w") as f:
    f.write(json.dumps(diagnostics_data, indent=4))

