from abc import ABCMeta, abstractmethod
from firedrake import *
import argparse

class InputParser(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="setup shallow water solver")
        geometry = self.parser.add_mutually_exclusive_group()
        geometry.add_argument("--sphere", nargs=2, type=float)
        geometry.add_argument("--square", nargs=2, type=float)
        self.parser.add_argument("problem_name")

        self._add_args()
        self.args = self.parser.parse_args()
        self.make_mesh()

    @abstractmethod
    def _add_args():
        pass

    def make_mesh(self):
        if self.args.square is not None:
            L = self.args.square[0]
            n = int(self.args.square[1])
            print("setting up square mesh of length %s with n %s"%(L, n))
            self.mesh = PeriodicSquareMesh(n, n, L)
        elif self.args.sphere is not None:
            R = self.args.sphere[1]
            ref = int(self.args.sphere[0])
            print("setting up sphere mesh with radius %s and refinement level %s"%(R, ref))
            self.mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref, degree=3)
            global_normal = Expression(("x[0]", "x[1]", "x[2]"))
            self.mesh.init_cell_orientations(global_normal)
            self.outward_normals = CellNormal(self.mesh)
        else:
            print("Geometry not recognised")


class RexiArgparser(InputParser):

    def _add_args(self):
        self.parser.add_argument("t", type=float)
        self.parser.add_argument("h", type=float)
        self.parser.add_argument("M", type=int)
        self.parser.add_argument("--family", default="BDM")
        self.parser.add_argument("--degree", type=int, default=0)
        self.parser.add_argument("--direct_solve", action="store_true")


class ImplicitMidpointArgparser(InputParser):

    def _add_args(self):
        self.parser.add_argument("dt", type=float)
        self.parser.add_argument("tmax", type=float)
        self.parser.add_argument("--family", default="BDM")
        self.parser.add_argument("--degree", type=int, default=0)
