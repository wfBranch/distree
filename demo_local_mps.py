# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash & Markus

"""

import scipy as sp
import numpy as np
import numpy.random as rnd
import sys
import os
import distree as dst
import distree.schedulers as SL
import anytree as atr
import yaml
import logging
import datetime
import argparse
import matica
import evoMPS.tdvp_gen as tdvp
import pickle
import h5py
from evolve import evolve_mps
from find_branches import find_branches

class Meas():
    def __init__(self, N, Dmax):
        self.N = N
        self.Dmax = Dmax
        self.buf = []

    def get_shape(self):
        raise NotImplementedError

    def get_dtype(self):
        raise NotImplementedError

    def measure(self, state, t, **kwargs):
        raise NotImplementedError

    def measure_to_buffer(self, state, t, **kwargs):
        self.buf.append(self.measure(state, t, **kwargs))

    def get_buffer(self):
        return self.buf

    def clear_buffer(self):
        self.buf = []

class MeasTime(Meas):
    def get_shape(self):
        return ()

    def get_dtype(self):
        return np.float_

    def measure(self, state, t):
        return t

# Expectation values of spin operatorstate.
class MeasMag(Meas):
    def get_shape(self):
        return (self.N, len(matica.mag_ops))

    def get_dtype(self):
        return np.float_

    def measure(self, state, t, **kwargs):
        return [[state.expect_1s(ST, n).real for ST in matica.mag_ops]
                   for n in range(1, state.N+1)]

# Energy
class MeasH(Meas):
    def get_shape(self):
        return ()

    def get_dtype(self):
        return np.float_

    def measure(self, state, t, **kwargs):
        #return sp.sum([state.expect_2s(state.ham[n], n)
        #                  for n in range(1, state.N)]).real
        return state.H_expect.real

# Energy with respect to a different Hamiltonian
class MeasHheis(Meas):
    def __init__(self, N, Dmax, pars):
        super().__init__(N, Dmax)
        hamDH, tamTVI = get_hamiltonian_decomp(pars)
        self.ham = hamDH

    def get_shape(self):
        return ()

    def get_dtype(self):
        return np.float_

    def measure(self, state, t, **kwargs):
        return sp.sum([state.expect_2s(self.ham[n], n)
                          for n in range(1,state.N)]).real

# Entropy between left half (sites 1 through n) and right half
# (sites n+1 though N)
class MeasEntropies(Meas):
    def get_shape(self):
        return (self.N-1,)

    def get_dtype(self):
        return np.float_

    def measure(self, state, t, **kwargs):
        return [state.entropy(n) for n in range(1, state.N)]

# Squares of the Schmidt coefficients across the bond at the center
# of the middle of the lattice
class MeasSchmidtsMid(Meas):
    def get_shape(self):
        return (self.Dmax,)

    def get_dtype(self):
        return np.float_

    def measure(self, state, t, **kwargs):
        val = np.zeros((self.Dmax,), dtype=np.float_)
        coeffs = state.schmidt_sq(int(state.N/2)).real
        val[:len(coeffs)] = coeffs
        return val

# Squares of the Schmidt coefficients across all the bonds
class MeasSchmidts(Meas):
    def get_shape(self):
        return (self.Dmax, self.N+1)

    def get_dtype(self):
        return np.float_

    def measure(self, state, t, **kwargs):
        val = np.zeros((self.Dmax, self.N+1), dtype=np.float_)
        for n in range(self.N+1):
            coeffs = sp.sqrt(state.schmidt_sq(n).real).real
            val[:len(coeffs), n] = coeffs
        return val


class Distree_Demo(dst.Distree):
    @staticmethod
    def dump_yaml(data, path):
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    @staticmethod
    def load_yaml(path):
        with open(path, 'r') as f:
            data = yaml.load(f)
        return data

    def get_taskdata_path(self, task_id):
        return os.path.join(self.data_path, '%s.yaml' % task_id)

    def save_task_data(self, taskdata_path, data, task_id, parent_id):
        data.update(task_id=task_id, parent_id=parent_id)
        self.dump_yaml(data, taskdata_path)

    def load_task_data(self, taskdata_path):
        taskdata = self.load_yaml(taskdata_path)
        task_id = taskdata.get("task_id", None)
        parent_id = taskdata.get("parent_id", None)
        return taskdata, task_id, parent_id

    @staticmethod
    def branch_treepath(parent_treepath, branch_num):
        return parent_treepath + '/%u' % branch_num

    def get_measurement_path(self, task_id):
        path = os.path.join(self.data_path,"{}.h5".format(task_id))
        return path

    @staticmethod
    def get_last_measurement_time(taskdata):
        measurement_path = taskdata["measurement_path"]
        with h5py.File(measurement_path, "r") as h5file:
            try:
                last_time = max(h5file["t"])
            except ValueError:
                # Couldn't take max, because the dataset was empty.
                last_time = -np.inf
        return last_time

    def initialize_measurements(self, taskdata):
        # TODO Would be nice to have another set of parameters in taskdata,
        # that specifies what measurements to do, instead of this hardcoded
        # stuff. Consider putting them in a measurement_pars.yaml file.
        initial_pars = self.load_yaml(taskdata["initial_pars_path"])
        N = initial_pars["N"]
        Dmax = initial_pars["bond_dim"]
        return {'t': MeasTime(N, Dmax),
                'H': MeasH(N, Dmax),
                'H_Heis': MeasHheis(N, Dmax, initial_pars),
                'mag': MeasMag(N, Dmax),
                'bond_entropies': MeasEntropies(N, Dmax),
                'middle_schmidt_sqs': MeasSchmidtsMid(N, Dmax),
                'all_schmidt': MeasSchmidts(N, Dmax)}

    def initialize_measurement_data(self, taskdata, meas):
        measurement_path = taskdata["measurement_path"]
        with h5py.File(measurement_path, "a") as h5file:
            for mname, m in meas.items():
                shp = m.get_shape()
                h5file.create_dataset(mname, (0,*shp), maxshape=(None,*shp), dtype=m.get_dtype())

    @staticmethod
    def measure_data(state, t, meas):
        for mname, m in meas.items():
            m.measure_to_buffer(state, t)

    @staticmethod
    def save_measurement_data(state, taskdata, meas):
        measurement_path = taskdata["measurement_path"]
        with h5py.File(measurement_path, "a") as h5file:
            for mname, m in meas.items():
                dset = h5file[mname]
                buf = m.get_buffer()
                m.clear_buffer()
                off = dset.shape[0]
                dset.resize(off+len(buf), axis=0)
                for j in range(len(buf)):
                    dset[off + j, ...] = buf[j]

    def get_state_path(self, task_id, t):
        path = os.path.join(self.data_path, "{}_t{}.p".format(task_id, t))
        return path

    def store_state(self, state, **kwargs):
        path = self.get_state_path(**kwargs)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        return path

    def load_state(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        return state

    def should_branch(self, state, t, prev_branching_time, taskdata):
        # TODO This should be replaced with some more sophisticated check of
        # when to branch, based also on properties of the state.
        return t > 2 and t - prev_branching_time > 1

    def evolve_state(self, state, taskdata):
        pars = self.load_yaml(taskdata["time_evo_pars_path"])
        state, time_increment = evolve_mps(state, pars)
        return state, time_increment

    def run_task(self, taskdata_path):
        # The taskdata file contains everything needed to run the task.
        # Initial values, parameters, and so on.
        # It also contains the task_id, generated by the scheduler when the
        # task was scheduled, and the parent_id, which may be `None`.
        taskdata, task_id, parent_id = self.load_task_data(taskdata_path)
        state_paths = taskdata.get("state_paths", {})

        # Check if the dictionary of states at different times is empty.
        if not state_paths:
            logging.info("The task {} has no states in it, so we initialize.")
            initial_pars = self.load_yaml(taskdata["initial_pars_path"])
            s = initial_state(initial_pars, get_hamiltonian(initial_pars))
            s_path = dtree.store_state(s, t=0.0, task_id=task_id)
            state_paths[0.0] = s_path
            taskdata["state_paths"] = state_paths

        #Initialize the measurement objects
        meas = self.initialize_measurements(taskdata)

        # Check if there already is a file for storing measurements for the
        # state. If not, create it.
        if ("measurement_path" not in taskdata
                or not taskdata["measurement_path"]):
            measurement_path = self.get_measurement_path(task_id)
            taskdata["measurement_path"] = measurement_path
            self.initialize_measurement_data(taskdata, meas)

        # Check if the job already has been run to a point where it has
        # branched.
        has_children = "children" in taskdata and taskdata["children"]
        if has_children:
            # TODO Decide what to do here. Maybe ask the user to set some
            # command line parameter saying "Yes, please recreate children."
            msg = "This task already has children."
            raise NotImplementedError(msg)

        prev_checkpoint = max(state_paths)
        t = prev_checkpoint  # The current time in the evolution.
        state_path = state_paths[t]
        state = self.load_state(state_path)
        prev_branching_time = min(state_paths)
        prev_measurement_time = self.get_last_measurement_time(taskdata)

        while t < taskdata["t_max"]:
            state, t_increment = self.evolve_state(state, taskdata)
            t += t_increment
            # Many times increments are multiples of powers of ten, but adding
            # them up incurs small floating point errors. We counter this by
            # rounding. It keeps the logs and filenames prettier.
            t = np.around(t, decimals=13)

            if t - prev_measurement_time >= taskdata["measurement_frequency"]:
                self.measure_data(state, t, meas)
                prev_measurement_time = t

            if t - prev_checkpoint >= taskdata["checkpoint_frequency"]:
                state_paths[t] = self.store_state(state, t=t, task_id=task_id)
                self.save_measurement_data(state, taskdata, meas)
                prev_checkpoint = t

            if self.should_branch(state, t, prev_branching_time, taskdata):
                did_branch = self.branch(state, t, taskdata, task_id)
                if did_branch:
                    break
                else:
                    prev_branching_time = t

        # Always store the state at the end of the simulation.
        # TODO Fix the fact that this gets run even if t > t_max from the
        # start.
        state_paths[t] = self.store_state(state, t=t, task_id=task_id)
        self.save_measurement_data(state, taskdata, meas)

        # Save the final taskdata, overwriting the initial data file(s)
        # Note that the values in taskdata that have been modified, have been
        # modified in place.
        self.save_task_data(taskdata_path, taskdata, task_id, parent_id) 

    def branch(self, state, t, taskdata, task_id):
        # Try to branch.
        branch_pars = self.load_yaml(taskdata["branch_pars_path"])
        children, coeffs = find_branches(state, branch_pars)
        num_children = len(children)

        if num_children < 2:
            # No branching happened.
            return False

        taskdata['num_children'] = num_children
        parent_treepath = taskdata['parent_treepath']
        branch_num = taskdata["branch_num"]
        treepath = self.branch_treepath(parent_treepath, branch_num)

        # Create taskdata files for, and schedule, children
        for i, (child, child_coeff) in enumerate(zip(children, coeffs)):
            # child_id = self.sched.get_id()  # Instead of this, see below.
            child_id = "{}_c{}".format(task_id, i)
            child_state_path = dtree.store_state(child, t=t, task_id=child_id)
            child_taskdata = {
                'parent_id': task_id, 
                'parent_treepath': treepath,
                'branch_num': i, 
                't_max': taskdata["t_max"],
                'state_paths': {t: child_state_path},
                'coeff': child_coeff*taskdata["coeff"], 
                'measurement_frequency': taskdata["measurement_frequency"],
                'checkpoint_frequency': taskdata["checkpoint_frequency"],
                'initial_pars_path': taskdata["initial_pars_path"],
                'time_evo_pars_path': taskdata["time_evo_pars_path"],
                'branch_pars_path': taskdata["branch_pars_path"]
            }

            # This will add each child task to the log, and schedule them to be
            # run. How they are run is up to the scheduler.
            child_id, child_path = self.schedule_task(
                task_id, child_taskdata, task_id=child_id
            )
            # NOTE: We could add more child info to the parent taskdata here
        return True

# End of custom Distree subclass #


# Build an anytree from saved data by parsing the log file.
def build_tree(dtree):
    top = None
    r = atr.Resolver('name')
    with open(dtree.log_path, "r") as f:
        for line in f:
            task_id1, parent_id1, taskdata_path = line.strip().split("\t")
            taskdata, task_id2, parent_id2 = dtree.load_task_data(taskdata_path)
            assert task_id1 == str(task_id2)
            assert parent_id1 == str(parent_id2)

            parent_treepath = taskdata.get('parent_treepath', '')
            branch_num = taskdata.get('branch_num', 0)
            num_children = taskdata.get('num_children', 0)
            
            if top is None:
                # Check that this really is a top-level node
                assert parent_id2 == "" or parent_id2 is None

                top = atr.Node('%u' % branch_num, task_id=task_id2, 
                                parent_id=parent_id2, 
                                num_children=num_children, 
                                data=taskdata)
            else:
                # pnode = atr.search.find_by_attr(top, parent_id, 
                # name='task_id') # not optimal, but should never fail

                # should be efficient 
                # (alternatively, keep a node dictionary with id's as keys)
                pnode = r.get(top, parent_treepath)

                atr.Node('%u' % branch_num, parent=pnode, task_id=task_id2, 
                         parent_id=parent_id2, num_children=num_children, 
                         data=taskdata)

    return top


def setup_logging():
    # Set up logging, both to stdout and to a file.
    # First get the logger and set the level to INFO
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Figure out a name for the log file.
    filename = os.path.basename(__file__).replace(".py", "")
    datetime_str = datetime.datetime.strftime(datetime.datetime.now(),
                                              '%Y-%m-%d_%H-%M-%S')
    title_str = ('{}_{}'.format(filename, datetime_str))
    logfilename = "log/{}.log".format(title_str)
    i = 0
    while os.path.exists(logfilename):
        i += 1
        logfilename = "log/{}_{}.log".format(title_str, i)
    # Create a handler for the file.
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    filehandler = logging.FileHandler(logfilename, mode='w')
    filehandler.setLevel(logging.INFO)
    # Create a handler for stdout.
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.INFO)
    # Create a formatter object, to determine output format.
    fmt = "%(asctime)s %(levelname).1s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    # Make both stdout and file logging use this formatter.
    consolehandler.setFormatter(formatter)
    filehandler.setFormatter(formatter)
    # Set the logger to use both handlers.
    logger.addHandler(filehandler)
    logger.addHandler(consolehandler)


def parse_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('taskfile', type=str, nargs="?", default="")
    parser.add_argument('--child', dest='child', default=False,
                        action='store_true')
    parser.add_argument('--show', dest='show', default=False,
                        action='store_true')
    args = parser.parse_args()
    return args


def bloch_state(theta,phi):
    return [sp.cos(theta/2)+0j, sp.sin(theta/2)*sp.exp(1j*phi)]


def get_hamiltonian(pars):
    N = pars["N"]    # System size
    hamiltonian = pars["hamiltonian"]  # Name of the model
    if hamiltonian.strip().lower() == "double_heisenberg_ising":
        chi = pars["chi"]
        omega = pars["omega"]
        stiffness = pars["stiffness"]
        J = -N*stiffness
        ham = matica.ham_heisenberg_ising(J, omega, chi, N)
    else:
        msg = "Unknown Hamiltonian: {}".format(hamiltonian)
        raise NotImplementedError(msg)

    return ham


def get_hamiltonian_decomp(pars):
    N = pars["N"]    # System size
    hamiltonian = pars["hamiltonian"]  # Name of the model
    if hamiltonian.strip().lower() == "double_heisenberg_ising":
        chi = pars["chi"]
        omega = pars["omega"]
        stiffness = pars["stiffness"]
        J = -N*stiffness
        # The Hamiltonian is the sum of two parts:
        # The double Heisengberg chain,
        hamDH = matica.ham_heisenberg_ising(J, 0, 0, N)
        # the transverse-field Ising between the chains.
        hamTVI = matica.ham_heisenberg_ising(0, omega, chi, N)
    else:
        msg = "Unknown Hamiltonian: {}".format(hamiltonian)
        raise NotImplementedError(msg)

    return hamDH, hamTVI


def initial_state(pars, ham):
    qn = pars["qn"]  # Local state space dimension
    N = pars["N"]    # System size
    bond_dim = pars["bond_dim"]  # Bond dimension
    zero_tol = pars["zero_tol"]  # Tolerance in evoMPS
    sanity_checks = pars["sanity_checks"]  # Sanity checks in evoMPS
    auto_truncate = pars["auto_truncate"]  # Automatic truncation in evoMPS
    th1  = pars["theta1"]  # Angle for the initial state
    phi1 = pars["phi1"]    # Angle for the initial state
    th2  = pars["theta2"]  # Angle for the initial state
    phi2 = pars["phi2"]    # Angle for the initial state

    D = [bond_dim]*(N+1)
    q = [qn]*(N+1)
    s = tdvp.EvoMPS_TDVP_Generic(N, D, q, ham)
    s.zero_tol = zero_tol
    s.sanity_checks = sanity_checks

    # "InitCondC" (ZY):
    # Comfirmed ZY trajectory chaotic in Mathematica with chi = 2
    init_state = sp.kron(bloch_state(th1, phi1), bloch_state(th2, phi2))
    s.set_state_product([init_state]*N)
    s.update(auto_truncate=auto_truncate)
    return s


if __name__ == "__main__":
    # Parse command line arguments.
    args = parse_args()
    # setup_logging is for the logging module, that is concerned with text
    # output (think print statements). It has nothing to do with the log file
    # of the Distree.
    setup_logging()
    # This log file keeps track of the tree.
    logfile = "./log/distreelog.txt"
    data_path = "./data/"
    # Create a scheduler and tell it what script to run to schedule tasks.
    sched = SL.Sched_Local(sys.argv[0], scriptargs=['--child'])

    # Create the tree object, telling it where the logfile lives, where the
    # taskdata files are to be stored, and giving it the scheduler to use.
    dtree = Distree_Demo(logfile, data_path, sched)

    # NOTE: This script is designed so that it can schedule the root job and
    # also child jobs, depending on the supplied command-line arguments.
    if args.show:
        # Print the tree from saved data
        top = build_tree(dtree)
        logging.info(atr.RenderTree(top))
    elif args.child:
        # Assume the first argument is a taskdata file for a child job.
        # This means the task should be run in the current process,
        # rather than be scheduled for later.
        dtree.run_task(args.taskfile)
    elif args.taskfile:
        # Assume the argument is a taskdata file to be used for a root job
        dtree.schedule_task_from_file(args.taskfile)
    else:
        # Save a simple initial taskdata file and schedule a root job.
        init_task_data = {
            'parent_id': None, 
            'parent_treepath': '',
            'branch_num': 0, 
            't_max': 7, 
            'coeff': 1.0,
            'measurement_frequency': 2,
            'checkpoint_frequency': 4,
            'initial_pars_path': 'confs/initial_pars.yaml',
            'time_evo_pars_path': 'confs/time_evo_pars.yaml',
            'branch_pars_path': 'confs/branch_pars.yaml'
        }
        # The following schedules a job (it will be run in a different process)
        dtree.schedule_task(None, init_task_data)
