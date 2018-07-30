# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import scipy as sp
import numpy as np
import numpy.random as rnd
import sys
import distree as dst
import distree.schedulers as SL
import anytree as atr

class Distree_Demo(dst.Distree):
    def get_taskdata_path(self, task_id):
        return self.data_path + '%s.npy' % task_id

    #TODO: Use a better-structured data format
    def save_task_data(self, taskdata_path, data):
        dlist = [data['parent_path'],
                data['branch_num'],
                data['t_0'],
                data['t_1'],
                data['t_max'],
                data['state'],
                data['coeff'],
                data['num_children'],
                data['task_id'],
                data['parent_id']
                ]
        sp.save(taskdata_path, sp.array(dlist, dtype=object))


    def load_task_data(self, taskdata_path):
        d = sp.load(taskdata_path)
        return {'parent_path': d[0], 
                'branch_num': d[1], 
                't_0': d[2], 
                't_1': d[3], 
                't_max': d[4], 
                'state': d[5],
                'coeff': d[6],
                'num_children': d[7],
                'task_id': d[8],
                'parent_id': d[9]}


    def branch_path(self, parent_path, branch_num):
        return parent_path + '/%u' % branch_num

    def run_task(self, taskdata_path):
        #Load data saved by parent task
        taskdata = self.load_task_data(taskdata_path)
        
        task_id = taskdata['task_id']
        parent_path = taskdata['parent_path']
        branch_num = taskdata['branch_num']
        t_0 = taskdata['t_0']
        t_max = taskdata['t_max']
        state = taskdata['state']
        coeff = taskdata['coeff']
        
        #Do the task
        t_1 = t_0 + 1
        state = sp.rand(4) * state

        if t_1 < t_max:
            num_children = 2 #rnd.randint(0, 4)
        else:
            num_children = 0

        #Save task output data
        taskdata['t_1'] = t_1
        taskdata['state'] = state
        taskdata['num_children'] = num_children
        self.save_task_data(taskdata_path, taskdata) #overwrites initial data

        branch_path = self.branch_path(parent_path, branch_num)

        #create init files for and schedule children
        for child_branch in range(num_children):
            child_taskdata = {'parent_path': branch_path, 
                                'branch_num': child_branch, 
                                't_0': t_1, 
                                't_1': None, 
                                't_max': t_max,
                                #a new state for each child
                                'state': state * (child_branch+1),
                                'coeff': coeff/(child_branch+1), 
                                'num_children': None,
                                #these ids are set by the scheduler
                                'task_id': None,
                                'parent_id': None}

            self.new_task(task_id, child_taskdata)

def build_tree(dtree):
    top = None
    r = atr.Resolver('name')
    with open(dtree.log_path, "r") as f:
        for line in f:
            task_id, parent_id, taskdata_path = line.strip().split("\t")
            taskdata = dtree.load_task_data(taskdata_path)
            parent_path = taskdata['parent_path']
            branch_num = taskdata['branch_num']
            num_children = taskdata['num_children']
            if top is None:
                assert parent_id == '' or parent_id == 'None'
                top = atr.Node('%u' % branch_num, task_id=task_id, parent_id=parent_id, num_children=num_children, data=taskdata)
            else:
                #pnode = atr.search.find_by_attr(top, parent_id, name='task_id') #not optimal, but should never fail
                pnode = r.get(top, parent_path) #should be fairly efficient (alternatively, keep a node dictionary with id's as keys)
                atr.Node('%u' % branch_num, parent=pnode, task_id=task_id, parent_id=parent_id, num_children=num_children, data=taskdata)

    return top


sched = SL.Sched_Local(sys.argv[0], scriptargs=['--child'])
dtree = Distree_Demo('logfile.txt', '', sched)

if len(sys.argv) == 1:
    init_task_data = {'parent_path': '', 
                    'branch_num': 0, 
                    't_0': 0, 
                    't_1': None, 
                    't_max': 4, 
                    'state': sp.rand(4),
                    'coeff': 1.0,
                    'num_children': None,
                    'task_id': None,
                    'parent_id': None}
    dtree.save_task_data('root.npy', init_task_data)
    dtree.new_task_from_file('root.npy')

elif len(sys.argv) == 2:
    if sys.argv[1] == '--show':
        top = build_tree(dtree)
        print(atr.RenderTree(top))
    else:
        dtree.new_task_from_file(sys.argv[1])

elif len(sys.argv) == 3:
    if sys.argv[2] == '--child':
        dtree.run_task(sys.argv[1])
