# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import scipy as sp
import numpy as np

import subprocess
import uuid
import sys

class Sched():
    def get_id(self):
        return None

    def schedule_task(self, taskdata_path):
        pass

class Sched_Local_SameProc(Sched):
    task_list = []
    id_last_used = -1

    def __init__(self, runfunc):
        self.runfunc = runfunc

    def get_id(self):
        self.id_last_used += 1
        return self.id_last_used

    def schedule_task(self, taskdata_path):
        self.task_list.append(taskdata_path)

    def run(self):
        while len(self.task_list) > 0:
            self.runfunc(self.task_list.pop())


class Sched_Local(Sched):
    def __init__(self, scriptpath, scriptargs=[], python_command=sys.executable):
        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command

    def get_id(self):
        return uuid.uuid1()

    def schedule_task(self, taskdata_path):
        args = [self.python_command, self.scriptpath, taskdata_path] + self.scriptargs
        print('Running: ', args)
        subprocess.Popen(args)
