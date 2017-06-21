from firedrake import *
import frexi_sphere.diagnostics
from os import path
import json


class Timestepping(object):

    def __init__(self, dirname, fields, params, timestepper):
        filename = path.join(dirname, 'output.pvd')
        self.outfile = File(filename)
        self.fields = fields
        self.field_dict = {field.name(): field for field in fields}
        self.params = params
        self.timestepper = timestepper
        self.diagnostics_file = path.join(dirname, "diagnostics.json")
        self.diagnostics = frexi_sphere.diagnostics.Diagnostics()

    def setup_diagnostics(self):

        diagnostics_list = ['max', 'min', 'l2']
        self.diagnostics_dict = {
            name: getattr(self.diagnostics, name) for name in diagnostics_list}
        self.diagnostics_data = {}
        for name, field in self.field_dict.iteritems():
            self.diagnostics_data[name] = {}
            for diagnostic in diagnostics_list:
                self.diagnostics_data[name][diagnostic] = []

        self.diagnostics_data['time'] = []
        self.diagnostics_data['max_courant'] = []
        self.diagnostics_data['energy'] = []

    def run(self, dt, tmax, steady_state=False):

        # get initial fields
        u0 = self.field_dict['u']
        h0 = self.field_dict['h']

        # save initial conditions for computing errors and set up error fields
        V1 = u0.function_space()
        V2 = h0.function_space()
        if steady_state:
            u_init = Function(V1, name="u_init").assign(u0)
            u_err = Function(V1, name="u_err")
            self.field_dict['u_err'] = u_err
            h_init = Function(V2, name="h_init").assign(h0)
            h_err = Function(V2, name="h_err")
            self.field_dict['h_err'] = h_err

        # setup diagnostics
        self.setup_diagnostics()

        # make functions for timestepping
        u_out = Function(V1)
        h_out = Function(V2)
        t = 0.

        # dump initial fields
        self.outfile.write(*self.fields)

        # save initial diagnostics
        for fname, field in self.field_dict.iteritems():
            for dname, diagnostic in self.diagnostics_dict.iteritems():
                self.diagnostics_data[fname][dname].append(diagnostic(field))

        max_courant = self.diagnostics.max_courant_number(u0, dt)
        energy = self.diagnostics.energy(h0, u0, self.params.g)
        self.diagnostics_data['max_courant'].append(max_courant)
        self.diagnostics_data['energy'].append(energy)
        self.diagnostics_data['time'].append(t)

        # print some diagnostics to check things are going well
        print t, energy, max_courant

        # timestepping loop
        while t < tmax - 0.5*dt:

            self.timestepper.apply(u0, h0, u_out, h_out)
            u0.assign(u_out)
            h0.assign(h_out)
            if steady_state:
                u_err.assign(u_out - u_init)
                h_err.assign(h_out - h_init)

            self.outfile.write(*self.fields)
            t += dt

            for fname, field in self.field_dict.iteritems():
                for dname, diagnostic in self.diagnostics_dict.iteritems():
                    self.diagnostics_data[
                        fname][dname].append(diagnostic(field))
            max_courant = self.diagnostics.max_courant_number(u0, dt)
            energy = self.diagnostics.energy(h0, u0, self.params.g)
            self.diagnostics_data['max_courant'].append(max_courant)
            self.diagnostics_data['energy'].append(energy)
            self.diagnostics_data['time'].append(t)

            # print some diagnostics to check things are going well
            print t, energy, max_courant

        # dump diagnostics dictionary
        with open(self.diagnostics_file, "w") as f:
            f.write(json.dumps(self.diagnostics_data, indent=4))
