# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash
"""

import distree.schedulers as SL
from demo_local import *
import anytree as atr

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
dtree.schedule_task('', init_task_data)
dtree.sched.run()

top = build_tree(dtree)
print(atr.RenderTree(top))