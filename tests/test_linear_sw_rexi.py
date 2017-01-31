from firedrake import *
from frexi_sphere.rexi import RexiTimestep
from frexi_sphere.linear_sw_solver import ImplicitMidpointLinearSWSolver
import pytest


def run(dirname, prob):
    family = "BDM"
    degree = 0
    n = 64
    t = 0.1
    h = 0.2
    M = 32

    mesh = PeriodicSquareMesh(n, n, 1.)

    dt = 0.01

    im = ImplicitMidpointLinearSWSolver(mesh, family, degree, prob, dt, dirname)
    im.run(t)
    im_h = im.h_end
    im_u = im.u_end
    r = RexiTimestep(mesh, family, degree, prob, t, dirname)
    r.run(h, M, True)
    rexi_h = r.h1r
    rexi_u = r.u1r
    h_err = sqrt(assemble((rexi_h - im_h)*(rexi_h - im_h)*dx))/sqrt(assemble(im_h*im_h*dx))
    u_err = sqrt(assemble(inner(rexi_u-im_u, rexi_u-im_u)*dx))/sqrt(assemble(inner(im_u, im_u)*dx))
    return h_err, u_err

@pytest.mark.parametrize("problem", ["wave_scenario", "gaussian_scenario"])
def test_linear_sw_rexi(tmpdir, problem):
    dirname = str(tmpdir)
    h_err, u_err = run(dirname, problem)
    assert h_err < 0.01
    assert u_err < 0.006
