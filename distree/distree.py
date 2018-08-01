# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import filelock
import pathlib

class Distree():
    def __init__(self, log_path, data_path, scheduler):
        self.data_path = data_path
        self.log_path = log_path
        self.sched = scheduler
        # Make sure that, if data_path or log_path includes a directory, that
        # directory exists, creating it if necessary.
        pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    """
    Logs a task in a central log file. This is convenient, as it is easy to
    reconstruct the tree by just scanning through the log.
    """
    def write_log_entry(self, task_id, parent_id, data_fn):
        lock = filelock.FileLock(self.log_path + '.lock')
        with lock:
            f = open(self.log_path, "a")
            f.write('%s\t%s\t%s\n' % (task_id, parent_id, data_fn))
            f.close()

    def schedule_task_from_file(self, taskdata_path, parent_id=None):
        task_data, task_id, parent_id_ld = self.load_task_data(taskdata_path)
        if parent_id is None:
            parent_id = parent_id_ld
        
        return self.schedule_task(parent_id, task_data)

    """
    Logs and schedules a new task as a child of the task specified by
    `parent_id`. All data required by the task are in `task_data`,
    whose content is specified in a sub-class.
    """
    def schedule_task(self, parent_id, task_data):
        #Generate an id now, before the task is scheduled, so we can save it
        #in the taskdata file.
        task_id = self.sched.get_id()
        
        #Save taskdata file
        taskdata_path = self.get_taskdata_path(task_id)
        self.save_task_data(taskdata_path, task_data, task_id, parent_id)
        
        #Log centrally
        self.write_log_entry(task_id, parent_id, taskdata_path)

        #Call the scheduler to schedule the task (it may not run until later)
        self.sched.schedule_task(taskdata_path)

        return task_id, taskdata_path

    """
    Specifies how the taskdata filename is derived from the task_id.
    """
    def get_taskdata_path(self, task_id):
        assert False, "Example implementation!"
        return self.data_path + '%s.npy' % task_id

    """
    Specifies the taskdata format. The implementation must save
    the task_id and the parent_id. Otherwise it's up to the implementor.
    """
    def save_task_data(self, taskdata_path, data, task_id, parent_id):
        assert False, "Example implementation!"
        f = open(taskdata_path, "w")
        f.write('%s\t%s\n' % (task_id, parent_id))
        f.close()

    """
    Loads taskdata from a file somehow. Must load the task_id and parent_id
    and return them separately.
    """
    def load_task_data(self, taskdata_path):
        assert False, "Example implementation!"
        f = open(taskdata_path, "r")
        task_id, parent_id = f[0].split()
        f.close()
        taskdata = []
        return taskdata, task_id, parent_id

    """
    This is where the actual content goes! Actually run a task.
    """
    def run_task(self, taskdata_path):
        pass
