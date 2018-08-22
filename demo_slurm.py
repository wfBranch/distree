# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import pathlib
import sys
import os
import argparse
import logging

import scipy as sp
import numpy as np
import numpy.random as rnd

import distree as dst
import anytree as atr


def get_taskdata_path(data_path, task_id):
    # Make sure that, if data_path includes a directory, that
    # directory exists, creating it if necessary.
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    return data_path + '%s.npy' % task_id


def save_task_data(task_id, parent_id, data, data_path):
    taskdata_path = get_taskdata_path(data_path, task_id)
    dlist = [task_id, parent_id,
            data['parent_path'],
            data['branch_num'],
            data['t_0'],
            data['t_1'],
            data['t_max'],
            data['state'],
            data['coeff'],
            data['num_children']
            ]
    sp.save(taskdata_path, sp.array(dlist, dtype=object))
    return taskdata_path


def load_task_data(taskdata_path):
    d = sp.load(taskdata_path)
    task_id = d[0]
    parent_id = d[1]
    taskdata = {'parent_path': d[2], 
            'branch_num': d[3], 
            't_0': d[4], 
            't_1': d[5], 
            't_max': d[6], 
            'state': d[7],
            'coeff': d[8],
            'num_children': d[9]}
    return taskdata, task_id, parent_id


def extend_branch_path(parent_path, branch_num):
    return parent_path + '/%u' % branch_num


def run_task(dtree, taskdata_path, data_path):
    #Load data saved by parent task. The taskdata file contains everything
    #needed to run the task. Initial values, parameters, and so on.
    #It also contains the task_id, generated by the scheduler when the
    #task was scheduled, and the parent_id, which may be `None`.
    taskdata, task_id, parent_id = load_task_data(taskdata_path)
    
    #Put some data into local variables for convenience
    parent_path = taskdata['parent_path']
    branch_num = taskdata['branch_num']
    t_0 = taskdata['t_0']
    t_max = taskdata['t_max']
    state = taskdata['state']
    coeff = taskdata['coeff']
    
    #Do the task. This is just a stupid dummy task.
    #It 'evolves time' from t_0 to t_0 + 1 and modifies the state.
    t_1 = t_0 + 1
    state = sp.rand(4) * state

    #Update some of the taskdata structure based on the task we ran.
    taskdata['t_1'] = t_1
    taskdata['state'] = state

    #If the simulation is not over, there will be children!
    if t_1 < t_max:
        taskdata = branch(dtree, taskdata, task_id, parent_id, data_path)

    #Save the final taskdata, overwriting the initial data file(s)
    taskdata_path = save_task_data(task_id, parent_id, taskdata, data_path) 


def branch(dtree, taskdata, task_id, parent_id, data_path):
    parent_path = taskdata['parent_path']
    branch_num = taskdata['branch_num']
    t_1 = taskdata['t_1']
    t_max = taskdata['t_max']
    state = taskdata['state']
    coeff = taskdata['coeff']

    #determine number of children
    num_children = 2
    #modify the 
    taskdata['num_children'] = 2

    #determine the child's parent branch path 
    branch_path = extend_branch_path(parent_path, branch_num)

    #create taskdata files for, and schedule, children
    for child_branch in range(num_children):
        child_taskdata = {'parent_path': branch_path, 
                            'branch_num': child_branch, 
                            't_0': t_1, 
                            't_1': None, 
                            't_max': t_max,
                            #a new state for each child
                            'state': state * (child_branch+1),
                            'coeff': coeff/(child_branch+1), 
                            'num_children': None}

        child_id = task_id + "_c%u" % child_branch
        child_taskdata_path = save_task_data(child_id, task_id, child_taskdata, data_path) 
        #This will add the child task to the log and schedule it to be run. 
        #How they are run is up to the scheduler.

        dtree.schedule_task(child_id, task_id, child_taskdata_path)

    return taskdata


#Build an anytree from saved data by parsing the log file.
def build_tree(log_path):
    top = None
    r = atr.Resolver('name')
    with open(log_path, "r") as f:
        for line in f:
            task_id1, parent_id1, taskdata_path = line.strip().split("\t")
            taskdata, task_id2, parent_id2 = load_task_data(taskdata_path)
            assert task_id1 == str(task_id2)
            assert parent_id1 == str(parent_id2)

            parent_path = taskdata['parent_path']
            branch_num = taskdata['branch_num']
            num_children = taskdata['num_children']
            
            if top is None:
                #Check that this really is a top-level node
                assert parent_id2 == "" or parent_id2 is None

                top = atr.Node('%u' % branch_num, task_id=task_id2, 
                                parent_id=parent_id2, 
                                num_children=num_children, 
                                data=taskdata)
            else:
                #pnode = atr.search.find_by_attr(top, parent_id, 
                # name='task_id') #not optimal, but should never fail

                #should be efficient 
                # (alternatively, keep a node dictionary with id's as keys)
                pnode = r.get(top, parent_path) 

                atr.Node('%u' % branch_num, parent=pnode, task_id=task_id2, 
                         parent_id=parent_id2, num_children=num_children, 
                         data=taskdata)

    return top


def setup_logging():
    # Set up logging, both to stdout and to a file.
    # First get the logger and set the level to INFO
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


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

if __name__ == "__main__":
    # Parse command line arguments.
    args = parse_args()
    # setup_logging is for the logging module, that is concerned with text
    # output (think print statements). It has nothing to do with the log file
    # of the Distree.
    setup_logging()
    
    scriptpath = os.path.abspath(__file__) 
    datapath = os.path.abspath(os.path.dirname(__file__))
    # This log file keeps track of the tree.
    logfile = os.path.join(datapath, 'logfile.txt')

    # Create a scheduler and tell it what script to run to schedule tasks.
    dtree = dst.Distree_Slurm(
        logfile,
        scriptpath,
        'rrg-gvidal',
        scriptargs='--child',
        python_command='python',
        time='walltime=1:00:00',
        cpus_per_task=1,
        mem_per_cpu='256M'
    )

    root_id = "testjob"

    # NOTE: This script is designed so that it can schedule the root job and
    # also child jobs, depending on the supplied command-line arguments.
    if args.show:
        # Print the tree from saved data
        top = build_tree(logfile)
        logging.info(atr.RenderTree(top))
    elif args.child:
        # Assume the first argument is a taskdata file for a child job.
        # This means the task should be run in the current process,
        # rather than be scheduled for later.
        run_task(dtree, args.taskfile, datapath)
    elif args.taskfile:
        # Assume the argument is a taskdata file to be used for a root job
        dtree.schedule_task(root_id, None, args.taskfile)
    else:
        # Save a simple initial taskdata file and schedule a root job.
        init_task_data = {'parent_path': '', 
                        'branch_num': 0, 
                        't_0': 0, 
                        't_1': None, 
                        't_max': 4, 
                        'state': sp.rand(4),
                        'coeff': 1.0,
                        'num_children': None}

        taskdata_path = save_task_data(root_id, None, init_task_data, datapath)
        # The following schedules a job (it will be run in a different process)
        dtree.schedule_task(root_id, None, taskdata_path)
