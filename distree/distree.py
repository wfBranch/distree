# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:10:00 2018

@author: Ash

"""

import filelock

class Distree():
    def __init__(self, log_path, data_path, scheduler):
        self.data_path = data_path
        self.log_path = log_path
        self.sched = scheduler

    def write_log_entry(self, task_id, parent_id, data_fn):
        lock = filelock.FileLock(self.log_path + '.lock')
        with lock:
            f = open(self.log_path, "a")
            f.write('%s\t%s\t%s\n' % (task_id, parent_id, data_fn))
            f.close()

    def new_task_from_file(self, taskdata_path, parent_id=None):
        task_data = self.load_task_data(taskdata_path)
        if parent_id is None:
            parent_id = task_data['parent_id']
        
        return self.new_task(parent_id, task_data)

    def new_task(self, parent_id, task_data):
        task_id = self.sched.get_id()
        
        #makes sure ids to be saved in task file are correct
        task_data['task_id'] = task_id
        task_data['parent_id'] = parent_id

        #save task data file
        taskdata_path = self.get_taskdata_path(task_id)
        self.save_task_data(taskdata_path, task_data)
        
        #log centrally
        self.write_log_entry(task_id, parent_id, taskdata_path)

        self.sched.schedule_task(taskdata_path)

        return task_id, taskdata_path

    def get_taskdata_path(self, task_id):
        assert False, "Example implementation!"
        return self.data_path + '%s.npy' % task_id

    def save_task_data(self, taskdata_path, data):
        assert False, "Example implementation!"
        f = open(taskdata_path, "w")
        f.write('%s\t%s\n' % (data['task_id'], data['parent_id']))
        f.close()

    def load_task_data(self, taskdata_path):
        assert False, "Example implementation!"
        f = open(taskdata_path, "r")
        task_id, parent_id = f[0].split()
        f.close()
        return {'task_id': task_id, 'parent_id': parent_id}
        
    def run_task(self, taskdata_path):
        pass
