from firedrake import *
from frexi_sphere.rexi import RexiTimestep
from frexi_sphere.linear_sw_solver import ImplicitMidpointLinearSWSolver
import pytest


def run(dirname, prob, reduce_to_half):
    family = "BDM"
    degree = 0
    n = 64
    t = 0.1
    h = 0.2
    M = 32

    mesh = PeriodicSquareMesh(n, n, 1.)

    dt = 0.01

    im = ImplicitMidpointLinearSWSolver(mesh, family, degree, prob, dt, dirname=dirname)
    im.run(t)
    im_h = im.h_end
    im_u = im.u_end
    r = RexiTimestep(mesh, family, degree, prob, t, h, M, reduce_to_half=reduce_to_half, nonlinear=False, dirname=dirname)
    r.run(r.setup.u0, r.setup.h0, True)
    rexi_h = r.hout
    rexi_u = r.uout
    h_err = sqrt(assemble((rexi_h - im_h)*(rexi_h - im_h)*dx))/sqrt(assemble(im_h*im_h*dx))
    u_err = sqrt(assemble(inner(rexi_u-im_u, rexi_u-im_u)*dx))/sqrt(assemble(inner(im_u, im_u)*dx))
    return h_err, u_err

@pytest.mark.parametrize("problem", ["wave_scenario", "gaussian_scenario"])
@pytest.mark.parametrize("reduce_to_half", [True, False])
def test_linear_sw_rexi(tmpdir, problem, reduce_to_half):
    dirname = str(tmpdir)
    h_err, u_err = run(dirname, problem, reduce_to_half)
    assert h_err < 0.01
    assert u_err < 0.006
