from firedrake import *
import json
from rexi import RexiTimestep
from linear_sw_solver import ImplicitMidpointLinearSWSolver

family = "BDM"
degree = 0
n = 128
t = 1.

mesh = PeriodicSquareMesh(n, n, 1.)

dt = 0.01

for prob in ["wave_scenario", "gaussian_scenario"]:
    im = ImplicitMidpointLinearSWSolver(mesh, family, degree, prob, dt)
    im.run(t)
    im_h = im.h_end
    im_u = im.u_end
    r = RexiTimestep(mesh, family, degree, prob, t)
    err = {'h': {}, 'u': {}}
    for h in [2**p for p in range(-9, 3)]:
        err['h'][h] = {}
        err['u'][h] = {}
        for M in [2**q for q in range(0, 15)]:
            r.run(h, M, True)
            rexi_h = r.h1r
            rexi_u = r.u1r
            h_err = sqrt(assemble((rexi_h - im_h)*(rexi_h - im_h)*dx))/sqrt(assemble(im_h*im_h*dx))
            u_err = sqrt(assemble(inner(rexi_u-im_u, rexi_u-im_u)*dx))/sqrt(assemble(inner(im_u, im_u)*dx))
            err['h'][h][M] = h_err
            err['u'][h][M] = u_err
            print prob, h, M, h_err, u_err

    with open(prob+"_err.json", "w") as f:
        f.write(json.dumps(err, indent=4))
