# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

Notes:
* Currently, 
"""

import scipy as sp
import numpy as np
import numpy.random as rnd
import distree as dst
import distree.schedulers as SL

class Distree_Demo(dst.Distree):
    def get_taskdata_path(self, task_id):
        return self.data_path + '%s.npy' % task_id

    #TODO: Use a better-structured data format
    def save_task_data(self, taskdata_path, data, task_id, parent_id):
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


    def load_task_data(self, taskdata_path):
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


    def branch_path(self, parent_path, branch_num):
        return parent_path + '/%u' % branch_num

    def run_task(self, taskdata_path):
        #Load data saved by parent task
        taskdata, task_id, parent_id = self.load_task_data(taskdata_path)
        
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
        self.save_task_data(taskdata_path, taskdata, task_id, parent_id) #overwrites initial data

        branch_path = self.branch_path(parent_path, branch_num)

        #create init files for and schedule children
        for child_branch in range(num_children):
            child_taskdata = {'parent_path': branch_path, 
                                'branch_num': child_branch, 
                                't_0': t_1, 
                                't_1': None, 
                                't_max': t_max,
                                'state': state * (child_branch+1), #a new state for each child
                                'coeff': coeff/(child_branch+1), 
                                'num_children': None}

            self.new_task(task_id, child_taskdata)



def runfunc(taskdata_path):
    dtree.run_task(taskdata_path)

sched = SL.Sched_Local_SameProc(runfunc)

dtree = Distree_Demo('logfile.txt', '', sched)

init_task_data = {'parent_path': '', 
                    'branch_num': 0, 
                    't_0': 0, 
                    't_1': None, 
                    't_max': 4, 
                    'state': sp.rand(4),
                    'coeff': 1.0,
                    'num_children': None}
dtree.new_task('', init_task_data)
dtree.sched.run()

import anytree as atr

def build_tree(dtree):
    top = None
    r = atr.Resolver('name')
    with open(dtree.log_path, "r") as f:
        for line in f:
            task_id, parent_id, taskdata_path = line.strip().split("\t")
            taskdata, task_id2, parent_id2 = dtree.load_task_data(taskdata_path)
            assert task_id == str(task_id2)
            assert parent_id == str(parent_id2)

            parent_path = taskdata['parent_path']
            branch_num = taskdata['branch_num']
            num_children = taskdata['num_children']
            
            if top is None:
                assert parent_id == ""
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

top = build_tree(dtree)
print(atr.RenderTree(top))