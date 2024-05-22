import os
import csv
import pandas as pd
import multiprocessing
import threading
import numpy as np
import utils
import time
from clock import tic_svc



class GPU:
    def __init__(self,memory,utilization,dram,train_time,clock,ddl):
        # self.duration = duration
        self.memory = memory
        self.utilization = utilization
        self.dram = dram
        # self.taggpu = taggpu
        self.tasks = []
        self.finalscheduled = []
        self.latency = []
        self.taskrun = 0
        self.inferencetask_count = 0
        self.gpucount = 0
        self.clock = clock
        self.num_train = 0
        self.num_infer = 0
        # self.ddl = ddl
        self.train_time = train_time
        self.occupied = False
        self.ddl = ddl
        sample_task = {
            "duration": 10,
            "memory": 5,
            "utilization":100
        }
    
    # Used for removing the finished task from the GPU
    # Checks the current time and duration

    def remove_finished_tasks(self):
        current_time = self.clock.get_cur_time()
        task_to_pop = []
        diff=0
     
        for task in self.tasks:
            diff = current_time - task["start"]
            if diff >= task["duration"]:
                task_to_pop.append(task)

        for task in task_to_pop:
            self.memory += task["memory"]
            self.utilization += task["utilization"]
            #self.finalscheduled.append(task["Jobid"])
            # print(f"Task {task['slno']} finished execution in GPU {self.gpuid}")
            #self.latency.append({'Jobid':task["slno"],'Latency':diff})
            if(task["tagtask"]==1):
                diff = self.clock.get_cur_time()-task["start_time"]
                if diff > task["duration"]*1.02:
                    task["meet_ddl"] = False
                self.ddl.append(task["meet_ddl"])
                self.num_infer = self.num_infer-1
                self.finalscheduled.append(task["Jobid"])
                self.latency.append({'Jobid':task["slno"],'Latency':diff})
                self.inferencetask_count=self.inferencetask_count+1
            if(task["tagtask"]==0):
                self.num_train = self.num_train-1
                self.train_time.append(diff)
                # with open("data/train_time", 'a+') as out:
                #     out.write(str(diff)+ '\n')
            self.taskrun = self.taskrun+1
            self.tasks.remove(task)
            self.occupied = False
            #self.tasks.remove(task)

      
        
    def gpu_count(self):
        if(self.utilization!=100 and self.memory!=self.dram):
            self.gpucount=1
        else:
            self.gpucount=0


    # Assigns tasks for the GPU
    # Returns to the scheduler the avilablity

    def assign_task(self,gpuno, task: dict, start_time):
        self.gpuid=gpuno
        # self.remove_finished_tasks()
        task_memory = task["memory"]
        task_utilization=task["utilization"]
        task_tag=task["tagtask"]
        task["tag"]=task_tag
    

        # if start_time <= self.clock.get_cur_time():
        #     # if(self.taggpu!=task_tag):
        #     #     return False

#         if(self.utilization!=100):
#             return False

#         if(self.memory!=self.dram):
#             return False
        if self.occupied:
            return False

        if task_memory > self.memory:
            return False # not allocating
        if task_utilization > self.utilization:
            return False
        # else:
        #     return False

        if task_tag == 0:
            self.num_train = self.num_train+1
        else:
            self.num_infer = self.num_infer+1
        task["start"] = self.clock.get_cur_time()
        self.tasks.append(task)
        self.memory -= task_memory
        self.utilization -= task_utilization
        self.occupied = True
        return True

   

