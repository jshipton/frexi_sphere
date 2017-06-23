from firedrake import *
from frexi_sphere.exponential_integrators import LinearExponentialIntegrator
from frexi_sphere.timestepping import Timestepping
from frexi_sphere.linear_sw_solver import ImplicitMidpointLinearSWSolver
from frexi_sphere.sw_setup import SetupShallowWater
import pytest


def run(dirname, prob, reduce_to_half, direct):
    family = "BDM"
    degree = 0
    n = 64
    t = 0.1
    h = 0.2
    M = 32

    mesh = PeriodicSquareMesh(n, n, 1.)
    setup = SetupShallowWater(mesh, family, degree, prob)
    setup.ics()
    V1 = setup.spaces['u']
    V2 = setup.spaces['h']
    u0 = Function(V1,name="u").assign(setup.u0)
    h0 = Function(V2,name="h").assign(setup.h0)
    rexi_u = Function(V1)
    rexi_h = Function(V2)

    dt = 0.01

    im = ImplicitMidpointLinearSWSolver(setup, dt, dirname=dirname)
    im.run(t)
    im_h = im.h_end
    im_u = im.u_end
    r = LinearExponentialIntegrator(setup, t, direct, h, M, reduce_to_half=reduce_to_half)
    stepper = Timestepping(dirname, [u0, h0], setup.params, r)
    rexi_h, rext_u = stepper.run(t, t, return_end=True)
    h_err = sqrt(assemble((rexi_h - im_h)*(rexi_h - im_h)*dx))/sqrt(assemble(im_h*im_h*dx))
    u_err = sqrt(assemble(inner(rexi_u-im_u, rexi_u-im_u)*dx))/sqrt(assemble(inner(im_u, im_u)*dx))
    return h_err, u_err

@pytest.mark.parametrize("problem", ["wave_scenario", "gaussian_scenario"])
@pytest.mark.parametrize("reduce_to_half", [True, False])
@pytest.mark.parametrize("direct", [True, False])
def test_linear_sw_rexi(tmpdir, problem, reduce_to_half, direct):
    dirname = str(tmpdir)
    h_err, u_err = run(dirname, problem, reduce_to_half, direct)
    assert h_err < 0.01
    assert u_err < 0.006
