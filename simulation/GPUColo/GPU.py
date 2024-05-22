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
    def __init__(self,memory,utilization,dram,clock,ddl,train_time,slo_factor):
        # self.duration = duration
        self.memory = memory*1.0
        self.utilization = utilization*1.0
        self.dram = dram
        self.tasks = []
        self.finalscheduled = []
        self.latency = []
        self.avglat = []
        # self.sysid = sysid
        self.taskrun = 0
        self.gpucount = 0
        self.inferenceocount_sim = 0
        self.secheduledjobid = []
        self.new_slo = 0.05 # updated slo after getting a new inference task
        self.prev_slo = 0 # slo before getting a new inference task
        self.gpuflag = 5
        self.gpudefaultutilization = 100.0
        self.clock = clock
        self.num_infer = 0
        self.num_train = 0
        self.init_util = utilization
        self.new_iter_delay = 0.1
        self.slo_factor = slo_factor # relax SLO
        self.slo_factor_max = slo_factor # relax SLO
        self.gpu_time = clock.get_cur_time()
        self.ddl = ddl
        self.train_time = train_time
        self.control_sleep=0
        self.control_util=0
        self.gpuid = 0
        sample_task = {
            "duration": 10,
            "memory": 5,
            "utilization":100
        }
    
    # Used for removing the finished task from the GPU
    # Checks the current time and duration

    def remove_finished_tasks(self):
        # current_time = self.clock.get_cur_time()
        # print("\n Current time %d"%(self.clock.get_cur_time()))
        task_to_pop = []
        diff=self.clock.get_cur_time()-self.gpu_time
        self.gpu_time = self.clock.get_cur_time()
        for task in self.tasks:
            # diff = current_time - task["start"]
            if task["tagtask"] == 0: # train
                tic_amount = task["speed"]*diff
                task["iters"] = task["iters"]-tic_amount
                if task["iters"] <= 0:
                    self.num_train = self.num_train-1
                    assert self.num_train >= 0
                    task_to_pop.append(task)
            if task["tagtask"] == 1:
                tic_amount = task["speed"]*diff
                task["batches"] = task["batches"]-tic_amount
                if task["batches"] <= 0:
                    self.num_infer = self.num_infer-1
                    assert self.num_infer >= 0
                    task_to_pop.append(task)
                    
                    # if self.num_infer == 0:
                    #     for t in self.tasks:
                    #         if t["tagtask"] == 0:
                    #             t["duration"] = 0.9*t["duration"]+0.1*(self.clock.get_cur_time()-t["start"])

        for task in task_to_pop:
            self.memory += task["memory"]
            self.utilization += task["utilization"]
            # print(f"Task {task['slno']} finished execution in GPU {self.gpuid}")
            # diff = self.clock.get_cur_time()-task["start"]
            diff = self.clock.get_cur_time()-task["start_time"]
            # print("Task %d, Elapsed %d"%(task["slno"],diff))
            if(task["tagtask"] == 1):
                if diff > task["duration"]*1.02:
                    task["meet_ddl"] = False
                self.ddl.append(task["meet_ddl"])
                # print(task["meet_ddl"])
                # self.finalscheduled.append(task["Jobid"])
                # self.latency.append({'Jobid':task["slno"],'Latency':diff,'Utilization':self.utilization-task["utilization"],'taskstogether':len(task_to_pop)-1})
                # self.inferenceocount_sim=self.inferenceocount_sim+1
            if(task["tagtask"]==0):
                self.train_time.append(diff)
                # with open("data/train_time", 'a+') as out:
                    # out.write(str(diff)+ '\n')
            self.taskrun=self.taskrun+1
            self.tasks.remove(task)
        
        
        # if (self.num_train <= 1) and (self.num_infer < 3):
        sum_util = 0
        for t in self.tasks:
            sum_util = sum_util+t["utilization"]
        if sum_util > 100:
            self.utilization = 0
        else:
            self.utilization = 100-sum_util
#             if self.gpuid == 45:
#                 print("After removing tasks")
#                 print("Util %.2f"%self.utilization)
#                 for t in self.tasks:
#                     print("tag: %d, util %.2f, control util %.2f"%(t["tagtask"], t["utilization"], t["control_util"]))
            
#     def delcheck(self,ch):
#         delta=0
#         sleep=1
#         #print("Util = ",ch)
#         for i in range(0,len(self.sysid)):
#             if(ch==i):
#                 y=float(self.sysid[i])
#                 delta=delta+y*sleep        
#         return delta/5

    def gpu_count(self):
        if(self.utilization!=self.gpudefaultutilization and self.memory!=self.dram):
            self.gpucount=1
        else:
            self.gpucount=0
    
    # def slopecheck(self,ch):
    #     y=0
    #     for i in range(0,len(self.sysid)):
    #         if((ch-20)==i):
    #             y=float(self.sysid[i])
    #     # return y
    #     return y


    # Assigns tasks for the GPU
    # Returns to the scheduler the avilablity
    
    def assign_task(self,gpuno, task: dict, start_time):
        self.gpuid=gpuno
        # self.remove_finished_tasks()
        task_memory = task["memory"]
        task_utilization=task["utilization"]
        task_tag=task["tagtask"]
        # task["tag"]=task_tag
        
        # if start_time <= self.clock.get_cur_time():
        if task_memory > self.memory:
            # print("OUT OF MEMORY")
            return False # not allocating
        if task_utilization > 100:
            return False
        if (task_utilization>self.utilization):
            if (task_tag == 1) and (self.num_infer < 3) and (self.num_train == 1):
                return -2
            elif (task_tag == 0) and (self.num_infer <= 3) and (self.num_train == 0):
                return -2
            else:
                return False

        # if (task_utilization>100) or (task_utilization>self.utilization):
        #     # print("OUT OF UTIL")
        #     return False

        if (task_tag == 0) and (self.num_infer == 0):
            return -1
        if (task_tag == 1) and (self.num_train == 0):
            return -1
        # else:
        # if task["tagtask"]==1: # increase the number of inference
        #     # update new slo
        #     if self.clock.get_cur_time() > task["start_time"]+1:
        #         self.new_slo = 0.05*1.0
        return True # First round dispatching test
        for t in self.tasks:
            t["control_util"] = t["utilization"]
            if t["tagtask"] == 0:
                t["speed"] = 1/0.2 #iters/sec
            elif t["tagtask"] == 1:
                t["speed"] = 1/0.05 #20batches/s, 50ms/batch
                
        if task["tagtask"]==0: # add num of iters for train task
            self.num_train = self.num_train + 1
            task["iters"] = round(task["duration"]/0.2)
            task["speed"] = 1/0.2 #iters/sec
            task["control_start"] = self.clock.get_cur_time()
            task["control_util"] = task["utilization"]
            # print("Time %.1f, Train Num %d"%(self.clock.get_cur_time(), self.num_train))
        if task["tagtask"]==1: # increase the number of inference
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
        
        # infer_flag=False; train_flag=False;
        # infer_cnt=0;train_cnt=0
        # for t in self.tasks:
        #     if t["tagtask"]==1:
        #         infer_flag=True
        #         infer_cnt=infer_cnt+1
        #     if t["tagtask"]==0:
        #         train_flag=True
        #         train_cnt=train_cnt+1
        
        if len(self.tasks) > 1:
            # init_percent = 100/len(self.tasks) # max 100% assigned to each task
            # total_percent = self.init_util-self.utilization # assign util proportionally
            # assert total_percent >= 0
            infer_batch_time_total = 0
            for task in self.tasks:
            #     if self.clock.get_cur_time() == task["start"]:
            #         if self.utilization < self.init_util-100:
            #             task["control_util"] = task["utilization"]/total_percent*100
            #             if task["control_util"] < task["utilization"]:
            #                 task["speed"] = task["utilization"]/total_percent*task["speed"]
            #                 assert task["speed"]>0
            #         else:
            #             task["control_util"] = task["utilization"]
                if task["tagtask"] == 1:
                    infer_batch_time_total = infer_batch_time_total+1/task["speed"]
            if infer_batch_time_total > 0:
                self.prev_slo = infer_batch_time_total/self.num_infer

        # speed_list = ["ASchedule:", str(self.clock.get_cur_time())] 
        # amount_list = ["ASchedule:", str(self.clock.get_cur_time())]
        # for task in self.tasks:
        #     if task["tagtask"] == 0:
        #         string = "T:%.fms"%(1000/task["speed"])
        #         string2 = "T:%.2fiters"%(task["iters"])
        #     else:
        #         string = "I:%.fms"%(1000/task["speed"])
        #         string2 = "I:%.2fbs"%(task["batches"])
        #     speed_list.append(string)
        #     amount_list.append(string2)
        # print(speed_list)
        # print(amount_list)
        task["gpu_ind"] = gpuno
        return True
    
    def assign_task2(self,gpuno, task: dict, start_time):
        self.gpuid=gpuno
        # self.remove_finished_tasks()
        task_memory = task["memory"]
        task_utilization=task["utilization"]
        task_tag=task["tagtask"]
        # task["tag"]=task_tag

        
        # if start_time <= self.clock.get_cur_time():
        if task_memory > self.memory:
            # print("OUT OF MEMORY")
            return False # not allocating
        if (task_utilization>100) or (task_utilization>self.utilization):
            # print("OUT OF UTIL")
            return False
        # else:
        #     return False
        # if self.gpuid == 45:
        #     print("Assign2, GPU %d, Train %d, Infer %d, Util %.3f, Tasks:"%(self.gpuid,self.num_train, self.num_infer, self.utilization))
            
        for t in self.tasks:
            t["control_util"] = t["utilization"]
            if t["tagtask"] == 0:
                t["speed"] = 1/0.2 #iters/sec
            elif t["tagtask"] == 1:
                t["speed"] = 1/0.05 #20batches/s, 50ms/batch
                
        if task["tagtask"]==0: # add num of iters for train task
            self.num_train = self.num_train + 1
            task["iters"] = round(task["duration"]/0.2)
            task["speed"] = 1/0.2 #iters/sec
            task["control_start"] = self.clock.get_cur_time()
            task["control_util"] = task["utilization"]
            # print("Time %.1f, Train Num %d"%(self.clock.get_cur_time(), self.num_train))
        if task["tagtask"]==1: # increase the number of inference
            self.num_infer = self.num_infer + 1
            task["control_util"] = task["utilization"]
            task["batches"] = round(task["duration"]/0.05)
            task["speed"] = 1/0.05 # 20batches/s, 50ms/batch
            task["meet_ddl"] = True
            # update new slo
            if self.clock.get_cur_time() > task["start_time"]+1:
                new_slo = 1-(self.clock.get_cur_time()-task["start_time"])/task["duration"]
                if new_slo <=0:
                    new_slo = 0.01
                self.slo_factor = min(new_slo, self.slo_factor_max) # relax SLO
                # self.new_slo = 0.05*1.0
            # self.new_slo = (task["duration"]-self.new_slo)/self.num_infer
            # if self.prev_slo == 0:
            #     self.prev_slo = self.new_slo
            # print("Time %.1f, Infer Num %d"%(self.clock.get_cur_time(), self.num_infer))
            # self.controller_sleep.set_slo(self.new_slo)
            # self.controller_util.set_slo(self.new_slo)
            
        task["start"] = self.clock.get_cur_time()
        self.tasks.append(task)
        # if self.gpuid == 45:
        #     for t in self.tasks:
        #         print(t)
        self.memory -= task_memory
        self.utilization -= task_utilization
        assert self.memory >= 0
        assert self.utilization >= 0
        
        # infer_flag=False; train_flag=False;
        # infer_cnt=0;train_cnt=0
        # for t in self.tasks:
        #     if t["tagtask"]==1:
        #         infer_flag=True
        #         infer_cnt=infer_cnt+1
        #     if t["tagtask"]==0:
        #         train_flag=True
        #         train_cnt=train_cnt+1
        
        if len(self.tasks) > 1:
            # init_percent = 100/len(self.tasks) # max 100% assigned to each task
            # total_percent = self.init_util-self.utilization # assign util proportionally
            # assert total_percent >= 0
            infer_batch_time_total = 0
            for task in self.tasks:
                # if self.clock.get_cur_time() == task["start"]:
                #     if self.utilization < self.init_util-100:
                #         task["control_util"] = task["utilization"]/total_percent*100
                #         if task["control_util"] < task["utilization"]: 
                #             task["speed"] = task["utilization"]/total_percent*task["speed"]
                #             assert task["speed"]>0
                #     else:
                #         task["control_util"] = task["utilization"]
                if task["tagtask"] == 1:
                    infer_batch_time_total = infer_batch_time_total+1/task["speed"]
            if infer_batch_time_total > 0:
                self.prev_slo = infer_batch_time_total/self.num_infer

        # speed_list = ["ASchedule:", str(self.clock.get_cur_time())] 
        # amount_list = ["ASchedule:", str(self.clock.get_cur_time())]
        # for task in self.tasks:
        #     if task["tagtask"] == 0:
        #         string = "T:%.fms"%(1000/task["speed"])
        #         string2 = "T:%.2fiters"%(task["iters"])
        #     else:
        #         string = "I:%.fms"%(1000/task["speed"])
        #         string2 = "I:%.2fbs"%(task["batches"])
        #     speed_list.append(string)
        #     amount_list.append(string2)
        # print(speed_list)
        # print(amount_list)
        task["gpu_ind"] = gpuno
        return True

    def assign_task3(self,gpuno, task: dict, start_time):
        self.gpuid=gpuno
        # self.remove_finished_tasks()
        task_memory = task["memory"]
        task_utilization=task["utilization"]
        task_tag=task["tagtask"]
        # task["tag"]=task_tag

        
        # if start_time <= self.clock.get_cur_time():
        if task_memory > self.memory:
            # print("OUT OF MEMORY")
            return False # not allocating
        # else:
        if task_tag == 1:
            if (self.num_infer > 2) or (self.num_train > 1):
                return False
        else:
            if (self.num_infer > 3) or (self.num_train > 0):
                return False            
        # if self.gpuid == 45:
        #     print("Assign3, GPU %d, Train %d, Infer %d, Util %.3f, Tasks:"%(self.gpuid,self.num_train, self.num_infer, self.utilization))

                
        if task["tagtask"]==0: # add num of iters for train task
            existing_infer_util = 0
            for t in self.tasks:
                t["control_util"] = t["utilization"]
                t["speed"] = 1/0.05
                existing_infer_util = existing_infer_util+t["control_util"]
            self.num_train = self.num_train + 1
            task["iters"] = round(task["duration"]/0.2)
            task["control_start"] = self.clock.get_cur_time()
            task["control_util"] = 100-existing_infer_util
            task["speed"] = (1/0.2)*task["control_util"]/task["utilization"]
        if task["tagtask"]==1: # increase the number of inference
            existing_infer_util = 0
            for t in self.tasks:
                if t["tagtask"] == 1:
                    t["control_util"] = t["utilization"]
                    existing_infer_util = existing_infer_util+t["control_util"]
            for t in self.tasks:
                if t["tagtask"] == 0:
                    t["control_util"] = 100-existing_infer_util-task_utilization
                    assert task_utilization > 0
                    t["speed"] = (1/0.2)*t["control_util"]/task["utilization"]
            self.num_infer = self.num_infer + 1
            task["control_util"] = task["utilization"]
            task["batches"] = round(task["duration"]/0.05)
            task["speed"] = 1/0.05 # 20batches/s, 50ms/batch
            task["meet_ddl"] = True
            # update new slo
            if self.clock.get_cur_time() > task["start_time"]+1:
                new_slo = 1-(self.clock.get_cur_time()-task["start_time"])/task["duration"]
                if new_slo <=0:
                    new_slo = 0.01
                self.slo_factor = min(new_slo, self.slo_factor_max) # relax SLO
            # update new slo
            # if self.clock.get_cur_time() > task["start_time"]+1:
            #     self.new_slo = 0.05*1.0
            # self.new_slo = (task["duration"]-self.new_slo)/self.num_infer
            # if self.prev_slo == 0:
            #     self.prev_slo = self.new_slo
            # print("Time %.1f, Infer Num %d"%(self.clock.get_cur_time(), self.num_infer))
            # self.controller_sleep.set_slo(self.new_slo)
            # self.controller_util.set_slo(self.new_slo)
        task["start"] = self.clock.get_cur_time()
        self.tasks.append(task) 
        # if self.gpuid == 45:
        #     for t in self.tasks:
        #         print(t)
        #     print(" ")
           
        self.memory -= task_memory
        self.utilization = 0
        assert self.memory >= 0
        assert self.utilization >= 0
        
        # infer_flag=False; train_flag=False;
        # infer_cnt=0;train_cnt=0
        # for t in self.tasks:
        #     if t["tagtask"]==1:
        #         infer_flag=True
        #         infer_cnt=infer_cnt+1
        #     if t["tagtask"]==0:
        #         train_flag=True
        #         train_cnt=train_cnt+1
        
        if len(self.tasks) > 1:
            # init_percent = 100/len(self.tasks) # max 100% assigned to each task
            # total_percent = self.init_util-self.utilization # assign util proportionally
            # assert total_percent >= 0
            infer_batch_time_total = 0
            for task in self.tasks:
                # if self.clock.get_cur_time() == task["start"]:
                #     if self.utilization < self.init_util-100:
                #         task["control_util"] = task["utilization"]/total_percent*100
                #         if task["control_util"] < task["utilization"]: 
                #             task["speed"] = task["utilization"]/total_percent*task["speed"]
                #             assert task["speed"]>0
                #     else:
                #         task["control_util"] = task["utilization"]
                if task["tagtask"] == 1:
                    infer_batch_time_total = infer_batch_time_total+1/task["speed"]
            if infer_batch_time_total > 0:
                self.prev_slo = infer_batch_time_total/self.num_infer

        # speed_list = ["ASchedule:", str(self.clock.get_cur_time())] 
        # amount_list = ["ASchedule:", str(self.clock.get_cur_time())]
        # for task in self.tasks:
        #     if task["tagtask"] == 0:
        #         string = "T:%.fms"%(1000/task["speed"])
        #         string2 = "T:%.2fiters"%(task["iters"])
        #     else:
        #         string = "I:%.fms"%(1000/task["speed"])
        #         string2 = "I:%.2fbs"%(task["batches"])
        #     speed_list.append(string)
        #     amount_list.append(string2)
        # print(speed_list)
        # print(amount_list)
        task["gpu_ind"] = gpuno
        return True
    
    def invoke_control(self, enable_outer_loop):   
        if (self.num_infer >= 1) and (self.num_train >= 1):
            enable_control = True
            infer_batch_time_total = 0
            for task in self.tasks:
                # if task["speed"] == 0:
                #     for task in self.tasks:
                #         print(task)
                if task["tagtask"] == 1:
                    infer_batch_time_total = infer_batch_time_total+1/task["speed"] # for updating avg lat

            self.prev_slo = infer_batch_time_total/self.num_infer
            # if round(self.clock.get_cur_time(),0) % 60 == 0:
            # #     print(self.tasks)
            # print("\nControl Enabled, SLO %.2fms"%(self.new_slo*self.slo_factor*1000=))
        else:
            enable_control = False
        # if (self.num_infer>0) or (self.num_train>0):
        #     print("Time %d, InvC, num infer %d, num train %d"%(self.clock.get_cur_time(),self.num_infer,self.num_train))
        
        if enable_control:
            sleep_delta = -0.02*(self.new_slo*self.slo_factor-self.prev_slo)*1000 #-0.02
            self.control_sleep = self.control_sleep + sleep_delta

            change_util = False
            if self.control_sleep < 0:
                self.control_sleep = 0
                change_util = True
            if self.control_sleep > 1:
                self.control_sleep = 1
                change_util = True
            # if (round(self.clock.get_cur_time(),0) % 60 == 0):
            #     print("\nControl Enabled, SLO %.2fms, Sleep delta %fs\n"%(self.new_slo*self.slo_factor*1000, sleep_delta))
            
            # if round(sleep_delta,5) != 0:
            #     print("Sleep %f, Sleep Delta %f, SLO %.2fms"%(self.control_sleep, sleep_delta, self.new_slo*self.slo_factor*1000))

            # for task in self.tasks:
            #     if task["tagtask"] == 1:
            #         if task["duration"] > max_infer_t:
            #             max_infer_t = task["duration"]
            err_rate = (self.new_slo*self.slo_factor-self.prev_slo)/(self.new_slo*self.slo_factor)
            total_train_util = 0
            total_train_util2 = 0
            total_infer_util = 0
            total_infer_util2 = 0
            for task in self.tasks:
                if task["tagtask"] == 0:
#                     if (task["control_util"] == task["utilization"]):
#                         for task in self.tasks:
#                             print(task)
                            
#                     print("")
                    total_train_util = total_train_util+task["control_util"]
                    total_train_util2 = total_train_util2+task["utilization"]
                    # if total_train_util > 100:
                    # print("total %f"%(total_train_util))
                    # for t in self.tasks:
                    #     if t["tagtask"] == 0:
                    # if self.gpuid==11:
                    #     print("Boriginal %f, Bcontrol %f"%(task["utilization"], task["control_util"]))
                    
                if task["tagtask"] == 1:
                    total_infer_util = total_infer_util+task["control_util"]
                    total_infer_util2 = total_infer_util2+task["utilization"]
            released_portion = self.control_sleep/4 # sleep release from train, based on control period 4s
            # if total_train_util >= 100:
            #     print(total_train_util2)
            #     print(total_train_util)
            # assert total_train_util < 100
            # print(" ")
            # train_util_delta = 0
            
            for task in self.tasks: # training tasks
                if not change_util:
                    if task["tagtask"] == 1: # infer
                        task_released_portion = task["control_util"]/total_infer_util*(released_portion*total_train_util)
                     
                        # task["speed"] = (1/0.05)*(task_released_portion+task["control_util"])/task["utilization"]
                        task["speed"] = 1/(self.new_slo*self.slo_factor) 
                        assert task["speed"]>0
                        
                    elif task["tagtask"] == 0: # train
                        # task_released_portion = task["control_util"]/total_train_util*released_portion
                        task["speed"] = 1/(1/task["speed"]+self.control_sleep)
                        assert task["speed"]>0
                        
                if change_util: # saturate
                    if task["tagtask"] == 1: # infer
                        task_released_portion = task["control_util"]/total_infer_util*(released_portion*total_train_util)
                     
                        task["speed"] = (1/0.05)*(task_released_portion+task["control_util"])/task["utilization"]
                        # task["speed"] = 1/(self.new_slo*self.slo_factor) 
                        assert task["speed"]>0
                        
                    elif task["tagtask"] == 0: # train
                        # task_released_portion = task["control_util"]/total_train_util*released_portion
                        task["speed"] = 1/(1/task["speed"]+self.control_sleep)
                        if task["speed"] <= 0:
                            print("GPU No. %d, speed %.2f"%(self.gpuid, task["speed"]))
                            print(task)
                        assert task["speed"]>0       
                        
                if enable_outer_loop: # change_util
                    if task["tagtask"] == 0: # train
                        # delta_train = 0

                        if err_rate > 0: # increase training utilization, dec time
                            new_total_util = min(total_train_util+4.6*err_rate, 99) # 4.6
                            assert new_total_util <= 100
                        elif err_rate < 0:
                            new_total_util = max(total_train_util+4.8*err_rate, 1) #2.6
                            assert new_total_util > 0
                        elif err_rate == 0:
                            new_total_util = total_train_util
 
                        if total_train_util > 100:
                            print("total %f, new %f"%(total_train_util, new_total_util))
                            for task in self.tasks:
                                print(task)
                        assert total_train_util <= 100
                        assert new_total_util <= 100
        
                        new_util = task["utilization"]/total_train_util2*new_total_util
                        task["speed"] = (1/0.2)*new_util/task["utilization"]
                        # task["speed"] = 1/(0.5*(1/new_util-1/task["control_util"])+1/task["speed"])
                        assert task["speed"]>0
                        # if new_util < task["utilization"]:
                        # train_util_delta = train_util_delta + new_util-task["control_util"] # dec util
                        task["control_util"] = new_util
                        assert new_util > 0
                        # if round(task["control_util"], 6) == 76.974418:
                        #     print("here")
                        #     print("total %f, new %f"%(total_train_util, new_total_util))
                        #     print("task %f, total_train_util2 %f"%(task["utilization"], total_train_util2))
                        #     for t in self.tasks:
                        #         if t["tagtask"] == 0:
                        #             print("original %f, control %f"%(t["utilization"], t["control_util"]))
                        #     print("GPU ID %d"%(self.gpuid))
                        #     # exit()
                        # if self.gpuid==11:
                        #     print("Aoriginal %f, Acontrol %f"%(task["utilization"], task["control_util"]))
                        # elif new_util >= task["utilization"]:
                            # task["speed"] = 1/0.05
                            # train_util_delta = train_util_delta + task["utilization"]-task["control_util"] #inc util
                            # task["control_util"] = task["utilization"]
                            
                        # if round(self.clock.get_cur_time(),0) % 60 == 0:
                        #     print("MPS percentage changes, err_rate %.3f"%(err_rate*100.0))
                        
            if enable_outer_loop: #change_util, infer percentage change after enabling outer loop
                for task in self.tasks:
                    if task["tagtask"] == 1:
                        new_util = task["utilization"]/total_infer_util2*(100-new_total_util)
                        # if new_util < task["utilization"]:
                        task["speed"] = (1/0.05)*new_util/task["utilization"]
                        # task["speed"] = 1/(0.5*(1/new_util-1/task["control_util"])+1/task["speed"])
                        # else:
                        #     task["speed"] = 1/(self.new_slo*self.slo_factor)
                        # assert task["speed"] >= 0
                        task["control_util"] = new_util
                        if new_total_util == 100:
                            print(total_train_util)
                            print("train %d, infer %d"%(self.num_train, self.num_infer))
                            for task in self.tasks:
                                print(task)
                        assert new_util > 0
                        # if task["speed"] < 0:
                        #     task["speed"] = 0.1*task["control_util"]/total_infer_util
                        #     task["control_util"] = task["control_util"]/total_infer_util
                        # else:
                        #     task["control_util"] = task["control_util"]-train_util_delta*task["control_util"]/total_infer_util
        # recover training utilization after inference is done
        if (self.num_infer == 0) and (self.num_train > 0):
            for task in self.tasks:
                if task["tagtask"]==0:
                    task["speed"] = 1/0.2
                    task["control_util"] = task["utilization"]
                        
        if (self.num_train == 0) and (self.num_infer > 0):
            for task in self.tasks:
                if task["tagtask"]==1:
                    task["speed"] = 1/0.05
                    task["control_util"] = task["utilization"]

        if self.num_infer > 0:
            self.avglat.append(self.prev_slo)
            
        return 
    
    def admit_migrate(self, task):
        if task['memory'] > self.memory:
            # print("OUT OF MEMORY")
            return False # not allocating
        if task['utilization'] > self.utilization:
            # print("OUT OF UTIL")
            return False
        self.memory -= task['memory']
        self.utilization -= task['utilization']
        assert self.memory >= 0
        assert self.utilization >= 0
        task["gpu_ind"] = self.gpuid
        self.tasks.append(task)
        return True
    
    def migrate(self, task):
        self.tasks.remove(task)
        self.memory += task['memory']
        self.utilization += task['utilization']
        assert self.memory <= 32
        assert self.utilization <= 100

