import os
import csv
import pandas as pd
import multiprocessing
import threading
import numpy as np
import utils
import time
import math
from clock import tic_svc


class GPU:
    def __init__(self,memory,utilization,dram,train_time,clock,ddl):
        # self.duration=duration
        self.memory=memory
        self.utilization=utilization
        self.init_util=utilization
        self.dram=dram
        self.tasks = []
        self.finalscheduled=[]
        self.latency=[]
        # self.sysid=sysid
        self.taskrun=0
        self.gpucount=0
        self.inferenceocount_sim=0
        self.secheduledjobid=[]
        self.slo=44
        self.slo_factor=1.0
        self.gpuflag=5
        self.gpudefaultutilization=100
        self.clock = clock
        # self.taggpu = taggpu
        self.num_infer = 0
        self.num_train = 0
        self.train_time = train_time
        self.ddl = ddl
        self.gpu_time = clock.get_cur_time()
        # self.gpu_tag = gpu_tag
        sample_task = {
            "duration": 10,
            "memory": 5,
            "utilization":100
        }
    
    # Used for removing the finished task from the GPU
    # Checks the current time and duration

    def remove_finished_tasks(self):
        # print("train %d, infer %d"%(self.num_train,self.num_infer))
        # print(self.num_train)
        assert (len(self.tasks)==self.num_infer) or (len(self.tasks)==self.num_train)
        # current_time = self.clock.get_cur_time()
        # print("\n Current time %d"%(self.clock.get_cur_time()))
        task_to_remove = []
        diff=self.clock.get_cur_time()-self.gpu_time
        self.gpu_time = self.clock.get_cur_time()
        for task in self.tasks:
            # diff = current_time - task["start"]
            if task["tagtask"] == 0: # train
                tic_amount = task["speed"]*diff
                task["iters"] = task["iters"]-tic_amount
                if True:# if task["iters"] <= 0:
                    self.num_train = self.num_train-1
                    assert self.num_train >= 0
                    # task_to_pop.append(task)
                    self.memory += task["memory"]
                    self.utilization += task["utilization"]
                    self.taskrun=self.taskrun+1
                    task_to_remove.append(task)
                    self.train_time.append(self.clock.get_cur_time()-task["start"]+(task["start"]-task["start_time"])//10+round(task["iters"]/task["speed"]))
                    
            if task["tagtask"] == 1:
                tic_amount = task["speed"]*diff
                task["batches"] = task["batches"]-tic_amount
                if task["batches"] <= 0:
                    self.num_infer = self.num_infer-1
                    assert self.num_infer >= 0
                    # task_to_pop.append(task)
                    self.memory += task["memory"]
                    self.utilization += task["utilization"]
                    if task["start"]-task["start_time"]>=10:
                        diff_infer = self.clock.get_cur_time()-task["start"]+(task["start"]-task["start_time"])//10+round(task["batches"]/task["speed"])
                    else:
                        diff_infer = self.clock.get_cur_time()-task["start"]+round(task["batches"]/task["speed"])
                    if diff_infer > task["duration"]+0.01:
                        task["meet_ddl"] = False
                    self.ddl.append(task["meet_ddl"])
                    self.taskrun=self.taskrun+1
                    task_to_remove.append(task)
                    
                    # if self.num_infer == 0:
                    #     for t in self.tasks:
                    #         if t["tagtask"] == 0:
                    #             t["duration"] = 0.9*t["duration"]+0.1*(self.clock.get_cur_time()-t["start"])

        for task in task_to_remove:
            self.tasks.remove(task)
#             self.memory += task["memory"]
#             self.utilization += task["utilization"]
#             # print(f"Task {task['slno']} finished execution in GPU {self.gpuid}")
#             # diff = self.clock.get_cur_time()-task["start"]
#             diff = self.clock.get_cur_time()-task["start_time"]
#             # print("Task %d, Elapsed %d"%(task["slno"],diff))
#             if(task["tagtask"] == 1):
#                 if diff > task["duration"]*1.02:
#                     task["meet_ddl"] = False
#                 self.ddl.append(task["meet_ddl"])
#                 self.finalscheduled.append(task["Jobid"])
#                 self.latency.append({'Jobid':task["slno"],'Latency':diff,'Utilization':self.utilization-task["utilization"],'taskstogether':len(task_to_pop)-1})
#                 self.inferenceocount_sim=self.inferenceocount_sim+1
#             if(task["tagtask"]==0):
#                 self.train_time.append(diff)
#                 # with open("data/train_time", 'a+') as out:
#                     # out.write(str(diff)+ '\n')
            
#             self.tasks.remove(task)
#             self.taskrun=self.taskrun+1

        # if self.num_infer == 0:
        #     if self.num_train > 0:
        #         # init_percent = 100/self.num_train
        #         total_percent = self.init_util-self.utilization # assign util proportionally
        #         for task in self.tasks:
        #             if task["tagtask"]==0:
        #                 # task["control_util"] = init_percent
        #                 # if init_percent < task["control_util"]:
        #                 if total_percent > 100:
        #                     task["speed"] = task["utilization"]/total_percent*(1/0.1)
        #                     assert task["speed"]>0
        #                 else:
        #                     task["speed"] = 1/0.1
        #     # if self.clock.get_cur_time() > 2489580.0 and self.num_train > 0 and self.num_train <= 2:
        #     #     print(self.tasks)
        # if self.num_train == 0:
        #     if self.num_infer > 0:
        #         # init_percent = 100/self.num_infer
        #         total_percent = self.init_util-self.utilization # assign util proportionally
        #         for task in self.tasks:
        #             if task["tagtask"]==1:
        #                 # task["control_util"] = init_percent
        #                 if total_percent > 100:
        #                     task["speed"] = task["utilization"]/total_percent*(1/0.05)
        #                     assert task["speed"]>0
        #                 else:
        #                     task["speed"] = 1/0.05
 
    def gpu_count(self):
        if(self.utilization!=100 and self.memory!=self.dram):
            self.gpucount=1
        else:
            self.gpucount=0

    # Assigns tasks for the GPU
    # Returns to the scheduler the avilablity

    def assign_task(self, gpuno, task: dict, start_time):
        self.gpuid=gpuno
        # self.remove_finished_tasks()
        task_memory = task["memory"]
        task_utilization = task["utilization"]
        task_tag = task["tagtask"]
        # task["tag"] = task_tag

        # if(self.taggpu!=task_tag):
        #     return False



        # if start_time <= self.clock.get_cur_time():
        if task_memory > self.memory:
            return False # not allocating
        if task_utilization>self.utilization:
            return False
        # if task_tag != self.gpu_tag:
            # print(task)
            # exit()
            # return -1
        # if task_tag == 0: # train
        #     if self.num_infer > 0:
        #         return False
        # if task_tag == 1: # infer
        #     if self.num_train > 0:
        #         return False
        # else:
        #     return False

        if task["tagtask"]==0: # add num of iters for train task
            self.num_train = self.num_train + 1
            task["iters"] = round(task["duration"]/0.2)
            task["speed"] = 1/0.2 #iters/sec
            task["control_start"] = self.clock.get_cur_time()
            task["control_util"] = task["utilization"]
            # print("Time %.1f, Train Num %d"%(self.clock.get_cur_time(), self.num_train))
        elif task["tagtask"]==1: # increase the number of inference
            self.num_infer = self.num_infer + 1
            # update new slo
            # self.new_slo = (task["duration"]-self.new_slo)/self.num_infer
            # if self.prev_slo == 0:
            #     self.prev_slo = self.new_slo
            task["control_util"] = task["utilization"]
            task["batches"] = round(task["duration"]/0.05)
            task["speed"] = 1/0.05 # 20batches/s, 50ms/batch
            task["meet_ddl"] = True
            # print("Time %.1f, Infer Num %d"%(self.clock.get_cur_time(), self.num_infer))
            # self.controller_sleep.set_slo(self.new_slo)
            # self.controller_util.set_slo(self.new_slo)
            
        task["start"] = self.clock.get_cur_time()
        self.tasks.append(task)
        self.memory -= task_memory
        self.utilization -= task_utilization
        assert self.memory >= 0
        assert self.utilization >= 0
        
        # if len(self.tasks) > 1:
        #     # init_percent = 100/len(self.tasks) # max 100% assigned to each task
        #     total_percent = self.init_util-self.utilization # assign util proportionally
        #     assert total_percent >= 0
        #     infer_batch_time_total = 0
        #     for task in self.tasks:
        #         # if self.clock.get_cur_time() == task["start"]:
        #         if self.utilization < self.init_util-100:
        #             task["control_util"] = task["utilization"]/total_percent*100
        #             if task["control_util"] < task["utilization"]:
        #                 if task['tagtask'] == 0:
        #                     task["speed"] = task["utilization"]/total_percent*(1/0.1)
        #                     assert task["speed"]>0
        #                 elif task['tagtask'] == 1:
        #                     task["speed"] = task["utilization"]/total_percent*(1/0.05)
        #                     assert task["speed"]>0
        #         else:
        #             task["control_util"] = task["utilization"]
        #             if task['tagtask'] == 0:
        #                 task["speed"] = 1/0.1
        #             elif task['tagtask'] == 1:
        #                 task["speed"] = 1/0.05
        return True

   

