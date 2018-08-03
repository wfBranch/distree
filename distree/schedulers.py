# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import os

import scipy as sp
import numpy as np

import subprocess
import uuid
import sys
import logging

from shlex import quote

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
    def __init__(self, scriptpath, scriptargs=[],
                 python_command=sys.executable):
        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command

    def get_id(self):
        return uuid.uuid1()

    def schedule_task(self, taskdata_path):
        args = ([self.python_command, self.scriptpath, taskdata_path]
                + self.scriptargs)
        logging.info('Running: {}'.format(args))
        subprocess.Popen(args)

class Sched_PBS(Sched):
    def __init__(self, qname, scriptpath, scriptargs='', python_command=sys.executable, res_list='', job_env=''):
        self.qname = qname
        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command
        self.res_list = res_list
        self.job_env = job_env
        self.jobname = os.path.basename(scriptpath)

    def get_id(self):
        return uuid.uuid1() #tasks are assigned UUID's based on the host that schedules them

    def schedule_task(self, taskdata_path):
        scmd = '%s %s %s %s' % (self.python_command, self.scriptpath, taskdata_path, self.scriptargs)

        qsub_cmd = 'qsub -N %s -q %s -k oe' % (quote(self.jobname), self.qname)
        if len(self.res_list) > 0:
            qsub_cmd = qsub_cmd + ' -l %s' % self.res_list

        if len(self.job_env) > 0:
            qsub_cmd = qsub_cmd + ' -v %s' % self.job_env

        cmd = 'echo %s | %s' % (quote(scmd), qsub_cmd)
        logging.info('Running: ', cmd)
        p = subprocess.run(cmd, shell=True)
        p.check_returncode()
