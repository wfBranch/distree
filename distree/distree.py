# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import os
import socket
import time
import filelock
import pathlib

import subprocess
import uuid
import sys
import logging

from shlex import quote

class Distree_Base():
    def __init__(self, log_path, canary_path=''):
        self.log_path = log_path
        self.canary_path = canary_path

        # Make sure that, if the paths include a directory, that
        # directory exists, creating it if necessary.
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if canary_path:
            pathlib.Path(canary_path).parent.mkdir(parents=True, exist_ok=True)

    """
    Logs a task in a central log file. This is convenient, as it is easy to
    reconstruct the tree by just scanning through the log.
    """
    def write_log_entry(self, task_id, parent_id, data_fn):
        if not parent_id and os.path.isfile(self.log_path):
            raise Exception(
                "Log file already exists while scheduling root job!"
            )

        lock = filelock.FileLock(self.log_path + '.lock')
        with lock:
            f = open(self.log_path, "a")
            f.write('%s\t%s\t%s\n' % (task_id, parent_id, data_fn))
            f.close()

    """
    Logs and schedules a new task as a child of the task specified by
    `parent_id`. The Base class just writes to the log file. Tasks
    will not actually be run!
    """
    def schedule_task(self, task_id, parent_id, taskdata_path):
        if not parent_id:
            self.create_canary()

        if not self.canary_is_alive():
            raise Exception("Tried to schedule a task with a dead canary.")

        #Log centrally
        self.write_log_entry(task_id, parent_id, taskdata_path)

    def create_canary(self):
        if self.canary_path:
            open(self.canary_path, 'w').close()
            logging.info("Created canary at {}".format(self.canary_path))

    def canary_is_alive(self):
        if self.canary_path:
            try:
                open(self.canary_path, 'r').close()
                return True
            except IOError:
                logging.warn("Canary {} has died!".format(self.canary_path))
                return False
        else:
            # We never had a canary in the first place, right?
            return True


class Distree_Local(Distree_Base):
    def __init__(self, log_path, scriptpath, scriptargs=[],
                 python_command=sys.executable, canary_path=''):
        super().__init__(log_path, canary_path=canary_path)

        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command

    def schedule_task(self, task_id, parent_id, taskdata_path):
        super().schedule_task(task_id, parent_id, taskdata_path)

        args = ([self.python_command, self.scriptpath, taskdata_path]
                + self.scriptargs)
        logging.info('Running: {}'.format(args))
        subprocess.Popen(args)


class Distree_PBS(Distree_Base):
    def __init__(self, log_path, scriptpath, qname, schedule_host,
                 scriptargs=[], python_command=sys.executable, precmd='',
                 res_list='', job_env='', working_dir=os.getcwd(),
                 canary_path='', working_dir_qsub=None, stream_dir=''):
        super().__init__(log_path, canary_path=canary_path)

        self.qname = qname
        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command
        self.precmd = precmd
        self.res_list = res_list
        self.job_env = job_env
        self.working_dir = working_dir
        self.working_dir_qsub = working_dir_qsub
        self.stream_dir = stream_dir
        self.schedule_host = schedule_host

        if working_dir_qsub:
            pathlib.Path(working_dir_qsub).mkdir(parents=True, exist_ok=True)

        if stream_dir:
            pathlib.Path(stream_dir).mkdir(parents=True, exist_ok=True)

    def schedule_task(self, task_id, parent_id, taskdata_path, 
                        stream_path=''):
        super().schedule_task(task_id, parent_id, taskdata_path)
        
        if self.precmd:
            scmd = '%s;' % self.precmd
        else:
            scmd = ''

        scmd += '%s %s %s %s' % (
            quote(self.python_command),
            quote(self.scriptpath),
            quote(taskdata_path),
            " ".join(map(quote, self.scriptargs))
        )

        jobname = task_id
        qsub_cmd = 'qsub -d %s -N %s -q %s' % (
            quote(self.working_dir), 
            quote(jobname), 
            quote(self.qname)
        )

        if self.res_list:
            qsub_cmd += ' -l %s' % quote(self.res_list)

        if self.job_env:
            qsub_cmd += ' -v %s' % quote(self.job_env)

        if self.stream_dir and not stream_path:
            stream_path = os.path.join(self.stream_dir, str(task_id))

        if stream_path:
            qsub_cmd += ' -o %s.o' % quote(stream_path)
            qsub_cmd += ' -e %s.e' % quote(stream_path)

        cmd = 'echo %s | %s' % (quote(scmd), qsub_cmd)

        if socket.gethostname() == self.schedule_host:
            # Run qsub directly
            p = subprocess.run(cmd, shell=True, cwd=self.working_dir_qsub)
            logging.info('Launched: %s' % cmd)
        else:
            # SSH to remote host
            ssh_cmd = ['ssh', self.schedule_host, cmd]
            p = subprocess.run(ssh_cmd, cwd=self.working_dir_qsub)
            logging.info('Launched: %s' % ssh_cmd)

        p.check_returncode()


class Distree_Slurm(Distree_Base):
    def __init__(self, log_path, scriptpath, account,
                scriptargs=[], python_command=sys.executable,
                cpus_per_task=1, mem_per_cpu='', time='',
                working_dir=os.getcwd(), canary_path='',
                stream_dir=''):
        super().__init__(log_path, canary_path=canary_path)

        self.account = account
        self.scriptpath = scriptpath
        self.scriptargs = scriptargs
        self.python_command = python_command
        self.time = time
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu = mem_per_cpu
        self.working_dir = working_dir
        self.stream_dir = stream_dir

        if stream_dir:
            pathlib.Path(stream_dir).mkdir(parents=True, exist_ok=True)

    def schedule_task(self, task_id, parent_id, taskdata_path, stream_path=''):
        super().schedule_task(task_id, parent_id, taskdata_path)

        scmd = '%s %s %s %s' % (
            quote(self.python_command),
            quote(self.scriptpath),
            quote(taskdata_path),
            " ".join(map(quote, self.scriptargs))
        )

        script = "#!/bin/sh\n" + scmd

        jobname = task_id
        sbatch_cmd = 'sbatch -A %s -J %s -c %u' % (
            quote(self.account),
            quote(jobname),
            self.cpus_per_task
        )

        if self.time:
            sbatch_cmd += ' -t %s' % quote(self.time)

        if self.mem_per_cpu:
            sbatch_cmd += ' --mem-per-cpu=%s' % quote(self.mem_per_cpu)

        if self.stream_dir and not stream_path:
            stream_path = os.path.join(self.stream_dir, str(task_id))

        if stream_path:
            sbatch_cmd += ' -o %s_%%j.out' % quote(stream_path)

        cmd = 'echo %s | %s' % (quote(script), sbatch_cmd)

        p = subprocess.run(cmd, shell=True)
        logging.info('Launched: %s' % cmd)

        p.check_returncode()
