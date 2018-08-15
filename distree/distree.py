# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import os
import filelock
import pathlib

import subprocess
import uuid
import sys
import logging

from shlex import quote

class Distree_Base():
    def __init__(self, log_path):
        self.log_path = log_path
        # Make sure that, if log_path includes a directory, that
        # directory exists, creating it if necessary.
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    """
    Logs a task in a central log file. This is convenient, as it is easy to
    reconstruct the tree by just scanning through the log.
    """
    def write_log_entry(self, task_id, parent_id, data_fn):
        if not parent_id and os.path.isfile(self.log_path):
            raise ValueError(
                "Log file already exists while scheduling root job!"
                )

        lock = filelock.FileLock(self.log_path + '.lock')
        with lock:
            f = open(self.log_path, "a")
            f.write('%s\t%s\t%s\n' % (task_id, parent_id, data_fn))
            f.close()

    """
    Logs and schedules a new task as a child of the task specified by
    `parent_id`.
    """
    def schedule_task(self, task_id, parent_id, taskdata_path):
        #Log centrally
        self.write_log_entry(task_id, parent_id, taskdata_path)

        #Call the scheduler to schedule the task (it might not run until later)
        raise NotImplementedError


class Distree_Local(Distree_Base):
    def __init__(self, log_path, scriptpath, scriptargs=[],
                 python_command=sys.executable):
        super().__init__(log_path)

        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command

    def schedule_task(self, task_id, parent_id, taskdata_path):
        #Log centrally
        self.write_log_entry(task_id, parent_id, taskdata_path)

        args = ([self.python_command, self.scriptpath, taskdata_path]
                + self.scriptargs)
        logging.info('Running: {}'.format(args))
        subprocess.Popen(args)


class Distree_PBS(Distree_Base):
    def __init__(self, log_path, scriptpath, qname,
                scriptargs=[], python_command=sys.executable, 
                res_list='', job_env='', working_dir=os.getcwd()):
        super().__init__(log_path)

        self.qname = qname
        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command
        self.res_list = res_list
        self.job_env = job_env
        self.working_dir = working_dir

    def schedule_task(self, task_id, parent_id, taskdata_path):
        #Log centrally
        self.write_log_entry(task_id, parent_id, taskdata_path)
        
        scmd = '%s %s %s %s' % (self.python_command, self.scriptpath, 
                                taskdata_path, " ".join(self.scriptargs))

        jobname = task_id
        qsub_cmd = 'qsub -d %s -N %s -q %s' % (quote(self.working_dir), 
                                                quote(jobname), 
                                                self.qname)
        if len(self.res_list) > 0:
            qsub_cmd = qsub_cmd + ' -l %s' % self.res_list

        if len(self.job_env) > 0:
            qsub_cmd = qsub_cmd + ' -v %s' % self.job_env

        cmd = 'echo %s | %s' % (quote(scmd), qsub_cmd)
        logging.info('Running: ', cmd)
        p = subprocess.run(cmd, shell=True)
        p.check_returncode()
