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
        self.delay_infer = 0
        self.infer_util = 0
        self.num_train = 0
        self.train_util = 0
        self.train_util_req = 0
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
        # if 100-self.utilization-self.train_util-self.infer_util < -1e-5:
        #     print("remove task: "+str(self.gpuid))
        #     print("overall util: "+str(self.utilization))
        #     print("infer util: "+str(self.infer_util))
        #     print("train util: "+str(self.train_util))
        #     print("num_train: "+str(self.num_train))
        #     print("num_infer: "+str(self.num_infer))
        #     print(self.tasks)
        # self.utilization=round(self.utilization, 5)
        self.train_util=round(self.train_util, 5)
        self.infer_util=round(self.infer_util, 5)
        self.utilization = 100-self.train_util-self.infer_util
        # assert 100-self.utilization-self.train_util-self.infer_util <1e-2
        # assert 100-self.utilization-self.train_util-self.infer_util >-1e-2
        # current_time = self.clock.get_cur_time()
        # print("\n Current time %d"%(self.clock.get_cur_time()))
        # task_to_pop = []
        diff=self.clock.get_cur_time()-self.gpu_time
        self.gpu_time = self.clock.get_cur_time()
        # if (self.gpu_time>=494) and (self.gpu_time<=560) and (self.gpuid==40):
        #     print("GPU time: "+str(self.gpu_time)+" Diff: "+str(diff))
        # if (self.gpu_time==536) and (self.gpuid==40):
        #     print(self.tasks)
        #     print("TOTAL NUM: "+str(len(self.tasks)))
        #     print("\n ")
        if self.delay_infer == 0:
            self.slo_factor = 1.0
        task_to_remove = []
        for task in self.tasks:
            # diff = current_time - task["start"]
            if task["tagtask"] == 0: # train
                tic_amount = task["speed"]*diff
                task["iters"] = task["iters"]-tic_amount
                if task["iters"] <= 0:
                    self.num_train = self.num_train-1
                    assert self.num_train >= 0
                    # task_to_pop.append(task)
                    self.memory += task["memory"]
                    # self.utilization += task["control_util"]
                    self.train_util -= task["control_util"]
                    self.train_util_req -= task["utilization"]
                    self.utilization = 100-self.train_util-self.infer_util
                    # self.utilization = round(self.utilization, 2)
                    # if (self.utilization < 1e-5) and (self.utilization > -1e-5):
                    #     self.utilization = 0
                    self.taskrun = self.taskrun + 1
                    task_to_remove.append(task)
                    self.train_time.append(self.clock.get_cur_time()-task["start"]+(task["start"]-task["start_time"])//10+round(task["iters"]/task["speed"]))
                    if (self.train_util < 1e-5) and (self.train_util > -1e-5):
                        self.train_util = 0
                    if (self.train_util_req < 1e-5) and (self.train_util_req > -1e-5):
                        self.train_util_req = 0
                    
                    if (100-self.utilization-self.train_util-self.infer_util < -1e-5) or (self.train_util<-1e-5):
                        print(self.tasks)
                        print(self.utilization)
                        print(self.train_util)
                        print(self.infer_util)
                    # assert self.train_util >= -1e-2
                    # assert self.utilization <= 100+1e-2
                    # assert self.train_util_req >= -1e-2
                    # assert 100-self.utilization-self.train_util-self.infer_util <1e-2
                    # assert 100-self.utilization-self.train_util-self.infer_util >-1e-2
                    
            if task["tagtask"] == 1:
                # if task["speed"]<1/0.05:
                #     print(task["speed"])
                # assert task["speed"]>-1e-2 #=1/0.05
                tic_amount = int(round(task["speed"]*diff))
                # assert tic_amount >= 20
                task["batches"] = task["batches"]-tic_amount
                if task["batches"] <= 0:
                    self.num_infer = self.num_infer-1
                    if task['delay']==True:
                        self.delay_infer -= 1
                    assert self.delay_infer >= 0
                    assert self.num_infer >= 0
                    # task_to_pop.append(task)
                    self.memory += task["memory"]
                    # self.utilization += task["control_util"]
                    self.infer_util -= task["control_util"]
                    if (self.infer_util < 1e-5) and (self.infer_util > -1e-5):
                        self.infer_util = 0
                    self.utilization = 100-self.train_util-self.infer_util
                    # if (self.utilization < 1e-5) and (self.utilization > -1e-5):
                    #     self.utilization = 0
                    if task["start"]-task["start_time"]>=10:
                        diff_infer = self.clock.get_cur_time()-task["start"]+(task["start"]-task["start_time"])//10+round(task["batches"]/task["speed"])
                    else:
                        diff_infer = self.clock.get_cur_time()-task["start"]+round(task["batches"]/task["speed"])
                    # if task["slno"]==163516:
                    #     print("batches: "+str(task["batches"]))
                    #     print("end time: "+str(self.clock.get_cur_time()))
                    #     print("diff_infer: "+str(diff_infer))
                    #     print(task)
                    if diff_infer > task["duration"]+0.01:
                        task["meet_ddl"] = False
                        # if diff-task["duration"]>=4:
                        #     print(" ")
                        #     print("delay: "+str(diff-task["duration"]))
                        #     print(task)
                        #     print("tic_amount: "+str(tic_amount))
                        
                    self.ddl.append(task["meet_ddl"])
                    self.taskrun=self.taskrun+1
                    task_to_remove.append(task)
                    # assert self.infer_util >= 0
                    # assert self.utilization <= 100+1e-2
                    # if task["slno"] == 263:
                    #     print("Time: "+str(self.clock.get_cur_time())+" Diff: "+str(diff_infer))
                    #     print(task)
                    # if 100-self.utilization-self.train_util-self.infer_util > 1e-5:
                    #     print("remove infer task")
                    #     print("overall util: "+str(self.utilization))
                    #     print("infer util: "+str(self.infer_util))
                    #     print("train util: "+str(self.train_util))
                    #     print("num_train: "+str(self.num_train))
                    #     print("num_infer: "+str(self.num_infer))
                    #     print(self.tasks)
                    #     print(task)
                    # assert 100-self.utilization-self.train_util-self.infer_util <1e-2
                    # assert 100-self.utilization-self.train_util-self.infer_util >-1e-2
                    # if self.num_infer == 0:
                    #     for t in self.tasks:
                    #         if t["tagtask"] == 0:
                    #             t["duration"] = 0.9*t["duration"]+0.1*(self.clock.get_cur_time()-t["start"])

        for task in task_to_remove:
            self.tasks.remove(task)
        #     self.memory += task["memory"]
        #     self.utilization += task["utilization"]
        #     # print(f"Task {task['slno']} finished execution in GPU {self.gpuid}")
        #     # diff = self.clock.get_cur_time()-task["start"]
        #     diff = self.clock.get_cur_time()-task["start_time"]
        #     # print("Task %d, Elapsed %d"%(task["slno"],diff))
        #     if(task["tagtask"] == 1):
        #         if diff > task["duration"]*1.02:
        #             task["meet_ddl"] = False
        #         self.ddl.append(task["meet_ddl"])
        #         # print(task["meet_ddl"])
        #         # self.finalscheduled.append(task["Jobid"])
        #         # self.latency.append({'Jobid':task["slno"],'Latency':diff,'Utilization':self.utilization-task["utilization"],'taskstogether':len(task_to_pop)-1})
        #         # self.inferenceocount_sim=self.inferenceocount_sim+1
        #     if(task["tagtask"]==0):
        #         self.train_time.append(diff)
        #         # with open("data/train_time", 'a+') as out:
        #             # out.write(str(diff)+ '\n')
        #     self.taskrun=self.taskrun+1
        #     self.tasks.remove(task)
        
        
        # if (self.num_train <= 1) and (self.num_infer < 3):
        if (self.num_infer == 0) and (self.num_train > 0):
            self.train_util = 0
            for task in self.tasks:
                if task["tagtask"]==0:
                    task["speed"] = 1/0.2
                    task["control_util"] = task["utilization"]
                    self.train_util += task["control_util"]
            self.utilization = 100 - self.train_util
            if self.utilization<0:
                print("\n")
                print(self.tasks)
            # assert self.utilization >= -1e-2
                        
        if (self.num_train == 0) and (self.num_infer > 0):
            self.infer_util = 0
            for task in self.tasks:
                if task["tagtask"]==1:
                    task["speed"] = 1/0.05
                    task["control_util"] = task["utilization"]
                    self.infer_util += task["control_util"]
            self.utilization = 100 - self.infer_util
            # assert self.utilization >= -1e-2
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
    
    def assign_task_train(self,gpuno, task: dict, start_time):
        self.gpuid=gpuno
        # self.remove_finished_tasks()
        task_memory = task["memory"]
        task_utilization=task["utilization"]
        task_tag=task["tagtask"]
        # task["tag"]=task_tag
        
        if task_tag != 0:
            return False
        # if start_time <= self.clock.get_cur_time():
        if task_memory > self.memory:
            # print("OUT OF MEMORY")
            return False # not allocating
        # if task_utilization > 100:
        #     return False
        if self.utilization < task_utilization: #1
            return False
        
        if self.train_util_req+task_utilization > 100:
            return False
        # if gpuno==31:
        #     print("train before")
        #     print("overall util: "+str(self.utilization))
        #     print("infer util: "+str(self.infer_util))
        #     print("train util: "+str(self.train_util))
        #     print("num_train: "+str(self.num_train))
        #     print("num_infer: "+str(self.num_infer))
        #     # print(self.tasks)
        #     # print(task)
        #     print(" ")
            
        alloc_util = min(self.utilization, task_utilization)
        task["iters"] = round(task["duration"]/0.2)
        task["speed"] = 1/0.2*alloc_util/task_utilization #iters/sec
        task["control_start"] = self.clock.get_cur_time()
        task["control_util"] = alloc_util
        task["start"] = self.clock.get_cur_time()
        task["gpu_ind"] = gpuno
        self.num_train = self.num_train + 1
        self.tasks.append(task)
        self.memory -= task_memory
        self.train_util += alloc_util
        self.train_util_req += task_utilization
        # self.utilization -= alloc_util
        self.utilization = 100-self.train_util-self.infer_util
        # if (self.utilization < 1e-5) and (self.utilization > -1e-5):
        #     self.utilization = 0
        assert self.memory >= 0
        
        # if round(100-self.utilization-self.train_util-self.infer_util, 2) < 1e-5:
        #     print("assign train: "+str(self.gpuid))
        #     print("overall util: "+str(self.utilization))
        #     print("infer util: "+str(self.infer_util))
        #     print("train util: "+str(self.train_util))
        #     print("alloc_util: "+str(alloc_util))
        #     print("num_train: "+str(self.num_train))
        #     print("num_infer: "+str(self.num_infer))
        #     print(self.tasks)
        #     # print(task)
        #     print(" ")
        # assert self.utilization >= -1e-2
        # assert round(100-self.utilization-self.train_util-self.infer_util, 2) <1e-2
        # assert round(100-self.utilization-self.train_util-self.infer_util, 2) >-1e-2
        # infer_flag=False; train_flag=False;
        # infer_cnt=0;train_cnt=0
        # for t in self.tasks:
        #     if t["tagtask"]==1:
        #         infer_flag=True
        #         infer_cnt=infer_cnt+1
        #     if t["tagtask"]==0:
        #         train_flag=True
        #         train_cnt=train_cnt+1

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
        # if gpuno==31:
        #     print("train")
        #     print("overall util: "+str(self.utilization))
        #     print("infer util: "+str(self.infer_util))
        #     print("train util: "+str(self.train_util))
        #     print("alloc_util: "+str(alloc_util))
        #     print("num_train: "+str(self.num_train))
        #     print("num_infer: "+str(self.num_infer))
        #     print(self.tasks)
        #     print(task)
        #     print(" ")
        return True
    
    def assign_task_infer(self,gpuno, task: dict, start_time):
        self.gpuid=gpuno
        # self.remove_finished_tasks()
        task_memory = task["memory"]
        task_utilization=task["utilization"]
        task_tag=task["tagtask"]
        # task["tag"]=task_tag

        if task_tag != 1:
            return False
        # if start_time <= self.clock.get_cur_time():
        if task_memory > self.memory:
            # print("OUT OF MEMORY")
            return False # not allocating
        # if (self.clock.get_cur_time() >= 86400) and (self.clock.get_cur_time() <= 172800):
        train_util_reserve = max(self.train_util*0.9, self.num_train)
        # else:
            # train_util_reserve = self.train_util
        if (task_utilization>100) or (task_utilization>100-self.infer_util-train_util_reserve): #-self.train_util
            # print("OUT OF UTIL")
            return False
        
        self.num_infer = self.num_infer + 1

        task["control_util"] = task["utilization"]
        task["batches"] = int(task["duration"]/0.05)
        task["total_batches"] = int(task["duration"]/0.05)
        task["speed"] = 1/0.05 # 20batches/s, 50ms/batch
        task["meet_ddl"] = True
        task["start"] = self.clock.get_cur_time()
        task["gpu_ind"] = gpuno
        self.tasks.append(task)
        self.memory -= task_memory
        
        self.infer_util += task["utilization"]

        if (self.num_train > 0) and (self.train_util > 100-self.infer_util):
            remain_util = 100 - self.infer_util
            self.train_util = remain_util
            for t in self.tasks:
                if t["tagtask"] == 0:
                    t["control_util"] = round(remain_util/self.num_train, 5)
                    t["speed"] = 1/0.2*t["control_util"]/t["utilization"] #iters/sec
        self.utilization = 100-self.train_util-self.infer_util
        # if (self.utilization < 1e-5) and (self.utilization > -1e-5):
        #     self.utilization = 0
            
        # if (self.utilization < 0) or (100-self.utilization-self.train_util-self.infer_util > 1e-5):
        #     print("\n")
        #     print(self.tasks)
        #     print(self.train_util)
        #     print(self.infer_util)
        #     print(self.utilization)
        assert self.memory >= 0
        # assert self.utilization >= -1e-2  
        # assert 100-self.utilization-self.train_util-self.infer_util <1e-2
        # assert 100-self.utilization-self.train_util-self.infer_util >-1e-2
        # else:
        #     return False
        # if self.gpuid == 45:
        #     print("Assign2, GPU %d, Train %d, Infer %d, Util %.3f, Tasks:"%(self.gpuid,self.num_train, self.num_infer, self.utilization))
        # if task["tagtask"]==0: # add num of iters for train task
        #     self.num_train = self.num_train + 1
        #     task["iters"] = round(task["duration"]/0.2)
        #     task["speed"] = 1/0.2 #iters/sec
        #     task["control_start"] = self.clock.get_cur_time()
        #     task["control_util"] = task["utilization"]
        #     # print("Time %.1f, Train Num %d"%(self.clock.get_cur_time(), self.num_train))
            
        # for t in self.tasks:
        #     t["control_util"] = t["utilization"]
        #     if t["tagtask"] == 0:
        #         t["speed"] = 1/0.2 #iters/sec
        #     elif t["tagtask"] == 1:
        #         t["speed"] = 1/0.05 #20batches/s, 50ms/batch
                

        # if task["tagtask"]==1: # increase the number of inference
        #     self.num_infer = self.num_infer + 1
        #     task["control_util"] = task["utilization"]
        #     task["batches"] = round(task["duration"]/0.05)
        #     task["speed"] = 1/0.05 # 20batches/s, 50ms/batch
        #     task["meet_ddl"] = True
        #     # update new slo
        #     if self.clock.get_cur_time() > task["start_time"]+1:
        #         new_slo = 1-(self.clock.get_cur_time()-task["start_time"])/task["duration"]
        #         if new_slo <=0:
        #             new_slo = 0.01
        #         self.slo_factor = min(new_slo, self.slo_factor_max) # relax SLO
                # self.new_slo = 0.05*1.0
            # self.new_slo = (task["duration"]-self.new_slo)/self.num_infer
            # if self.prev_slo == 0:
            #     self.prev_slo = self.new_slo
            # print("Time %.1f, Infer Num %d"%(self.clock.get_cur_time(), self.num_infer))
            # self.controller_sleep.set_slo(self.new_slo)
            # self.controller_util.set_slo(self.new_slo)
            

        # if self.gpuid == 45:
        #     for t in self.tasks:
        #         print(t)

        
        # infer_flag=False; train_flag=False;
        # infer_cnt=0;train_cnt=0
        # for t in self.tasks:
        #     if t["tagtask"]==1:
        #         infer_flag=True
        #         infer_cnt=infer_cnt+1
        #     if t["tagtask"]==0:
        #         train_flag=True
        #         train_cnt=train_cnt+1

        if self.clock.get_cur_time() - task["start_time"]>=10:
            # print(task)
            self.slo_factor = 0.5 # relax SLO
            self.delay_infer += 1
            task["delay"] = True
        # else:
        #     self.slo_factor = self.slo_factor_max # relax SLO
        if  self.num_infer > 0:
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
                #             assert task["speed"]>-1e-2
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
        # if gpuno==31:
        #     print("infer ")
        #     print("overall util: "+str(self.utilization))
        #     print("infer util: "+str(self.infer_util))
        #     print("train util: "+str(self.train_util))
        #     print("num_train: "+str(self.num_train))
        #     print("num_infer: "+str(self.num_infer))
        #     print(self.tasks)
        #     print(task)
        #     print(" ")
        return True

#     def assign_task3(self,gpuno, task: dict, start_time):
#         self.gpuid=gpuno
#         # self.remove_finished_tasks()
#         task_memory = task["memory"]
#         task_utilization=task["utilization"]
#         task_tag=task["tagtask"]
#         # task["tag"]=task_tag

        
#         # if start_time <= self.clock.get_cur_time():
#         if task_memory > self.memory:
#             # print("OUT OF MEMORY")
#             return False # not allocating
#         # else:
#         if task_tag == 1:
#             if (self.num_infer > 2) or (self.num_train > 1):
#                 return False
#         else:
#             if (self.num_infer > 3) or (self.num_train > 0):
#                 return False            
#         # if self.gpuid == 45:
#         #     print("Assign3, GPU %d, Train %d, Infer %d, Util %.3f, Tasks:"%(self.gpuid,self.num_train, self.num_infer, self.utilization))

                
#         if task["tagtask"]==0: # add num of iters for train task
#             existing_infer_util = 0
#             for t in self.tasks:
#                 t["control_util"] = t["utilization"]
#                 t["speed"] = 1/0.05
#                 existing_infer_util = existing_infer_util+t["control_util"]
#             self.num_train = self.num_train + 1
#             task["iters"] = round(task["duration"]/0.2)
#             task["control_start"] = self.clock.get_cur_time()
#             task["control_util"] = 100-existing_infer_util
#             task["speed"] = (1/0.2)*task["control_util"]/task["utilization"]
#         if task["tagtask"]==1: # increase the number of inference
#             existing_infer_util = 0
#             for t in self.tasks:
#                 if t["tagtask"] == 1:
#                     t["control_util"] = t["utilization"]
#                     existing_infer_util = existing_infer_util+t["control_util"]
#             for t in self.tasks:
#                 if t["tagtask"] == 0:
#                     t["control_util"] = 100-existing_infer_util-task_utilization
#                     assert task_utilization > 0
#                     t["speed"] = (1/0.2)*t["control_util"]/task["utilization"]
#             self.num_infer = self.num_infer + 1
#             task["control_util"] = task["utilization"]
#             task["batches"] = round(task["duration"]/0.05)
#             task["speed"] = 1/0.05 # 20batches/s, 50ms/batch
#             task["meet_ddl"] = True
#             # update new slo
#             if self.clock.get_cur_time() > task["start_time"]+1:
#                 new_slo = 1-(self.clock.get_cur_time()-task["start_time"])/task["duration"]
#                 if new_slo <=0:
#                     new_slo = 0.01
#                 self.slo_factor = min(new_slo, self.slo_factor_max) # relax SLO
#             # update new slo
#             # if self.clock.get_cur_time() > task["start_time"]+1:
#             #     self.new_slo = 0.05*1.0
#             # self.new_slo = (task["duration"]-self.new_slo)/self.num_infer
#             # if self.prev_slo == 0:
#             #     self.prev_slo = self.new_slo
#             # print("Time %.1f, Infer Num %d"%(self.clock.get_cur_time(), self.num_infer))
#             # self.controller_sleep.set_slo(self.new_slo)
#             # self.controller_util.set_slo(self.new_slo)
#         task["start"] = self.clock.get_cur_time()
#         self.tasks.append(task) 
#         # if self.gpuid == 45:
#         #     for t in self.tasks:
#         #         print(t)
#         #     print(" ")
           
#         self.memory -= task_memory
#         self.utilization = 0
#         assert self.memory >= 0
#         assert self.utilization >= -1e-2
        
#         # infer_flag=False; train_flag=False;
#         # infer_cnt=0;train_cnt=0
#         # for t in self.tasks:
#         #     if t["tagtask"]==1:
#         #         infer_flag=True
#         #         infer_cnt=infer_cnt+1
#         #     if t["tagtask"]==0:
#         #         train_flag=True
#         #         train_cnt=train_cnt+1
        
#         if len(self.tasks) > 1:
#             # init_percent = 100/len(self.tasks) # max 100% assigned to each task
#             # total_percent = self.init_util-self.utilization # assign util proportionally
#             # assert total_percent >= 0
#             infer_batch_time_total = 0
#             for task in self.tasks:
#                 # if self.clock.get_cur_time() == task["start"]:
#                 #     if self.utilization < self.init_util-100:
#                 #         task["control_util"] = task["utilization"]/total_percent*100
#                 #         if task["control_util"] < task["utilization"]: 
#                 #             task["speed"] = task["utilization"]/total_percent*task["speed"]
#                 #             assert task["speed"]>-1e-2
#                 #     else:
#                 #         task["control_util"] = task["utilization"]
#                 if task["tagtask"] == 1:
#                     infer_batch_time_total = infer_batch_time_total+1/task["speed"]
#             if infer_batch_time_total > 0:
#                 self.prev_slo = infer_batch_time_total/self.num_infer

#         # speed_list = ["ASchedule:", str(self.clock.get_cur_time())] 
#         # amount_list = ["ASchedule:", str(self.clock.get_cur_time())]
#         # for task in self.tasks:
#         #     if task["tagtask"] == 0:
#         #         string = "T:%.fms"%(1000/task["speed"])
#         #         string2 = "T:%.2fiters"%(task["iters"])
#         #     else:
#         #         string = "I:%.fms"%(1000/task["speed"])
#         #         string2 = "I:%.2fbs"%(task["batches"])
#         #     speed_list.append(string)
#         #     amount_list.append(string2)
#         # print(speed_list)
#         # print(amount_list)
#         task["gpu_ind"] = gpuno
#         return True
    
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
            if enable_outer_loop:
                if err_rate > 1e-5: # increase training utilization, dec time
                    new_total_util = min(total_train_util+4.6*err_rate, 99) # 4.6
                    # assert new_total_util <= 100
                elif err_rate < -1e-5:
                    new_total_util = max(total_train_util+4.8*err_rate, 1*self.num_train) #2.6
                    # assert new_total_util > -1e-2
                else:
                    new_total_util = total_train_util
                if self.infer_util-(new_total_util-total_train_util)<1:
                    new_total_util = total_train_util
                elif new_total_util-total_train_util > self.infer_util:
                    new_total_util = total_train_util+self.infer_util/2.0
                new_total_util = round(new_total_util, 5)
            for task in self.tasks: # training tasks
                if not change_util:
                    if task["tagtask"] == 1: # infer
                        task_released_portion = task["control_util"]/total_infer_util*(released_portion*total_train_util)
                     
                        # task["speed"] = (1/0.05)*(task_released_portion+task["control_util"])/task["utilization"]
                        task["speed"] = round(1/(self.new_slo*self.slo_factor), 2) 
                        # assert task["speed"]>=1/0.05
                        
                    elif task["tagtask"] == 0: # train
                        # task_released_portion = task["control_util"]/total_train_util*released_portion
                        task["speed"] = 1/(1/task["speed"]+self.control_sleep)
                        # assert task["speed"]>-1e-2
                        
                if change_util: # saturate
                    if task["tagtask"] == 1: # infer
                        task_released_portion = task["control_util"]/total_infer_util*(released_portion*total_train_util)
                     
                        task["speed"] = round((1/0.05)*(task_released_portion+task["control_util"])/task["utilization"],2)
                        # task["speed"] = 1/(self.new_slo*self.slo_factor)
                        # if (task["speed"]-1/0.05 < 1e-5) and (task["speed"]-1/0.05 > -1e-5):
                        #     task["speed"] = 1/0.05
                        # if task["speed"]<1/0.05:
                        #     print("infer speed: "+str(task["speed"]))
                        #     print("sleep: "+str(self.control_sleep))
                        # assert task["speed"]>-1e-2 #=1/0.05
                        
                    elif task["tagtask"] == 0: # train
                        # task_released_portion = task["control_util"]/total_train_util*released_portion
                        task["speed"] = 1/(1/task["speed"]+self.control_sleep)
                        if task["speed"] <= 0:
                            print("GPU No. %d, speed %.2f"%(self.gpuid, task["speed"]))
                            print(task)
                        # assert task["speed"]>-1e-2       
                        
                if enable_outer_loop: # change_util
                    if task["tagtask"] == 0: # train
                        # delta_train = 0
                        
                        if total_train_util > 100:
                            print("total %f, new %f"%(total_train_util, new_total_util))
                            for task in self.tasks:
                                print(task)
                        # assert total_train_util <= 100
                        # assert new_total_util <= 100
                        self.train_util = round(new_total_util, 5)
        
                        new_util = task["utilization"]/total_train_util2*new_total_util
                        task["speed"] = (1/0.2)*new_util/task["utilization"]
                        # task["speed"] = 1/(0.5*(1/new_util-1/task["control_util"])+1/task["speed"])
                        # assert task["speed"]>-1e-2
                        # if new_util < task["utilization"]:
                        # train_util_delta = train_util_delta + new_util-task["control_util"] # dec util
                        task["control_util"] = round(new_util, 5)
                        # assert new_util > -1e-2
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
                self.infer_util += total_train_util-new_total_util
                self.utilization = round(100-self.infer_util-self.train_util, 2)
                # if (self.utilization < 1e-5) and (self.utilization > -1e-5):
                #     self.utilization = 0
                if self.utilization<0:
                    print("overall util: "+str(self.utilization))
                    print("infer util before: "+str(self.infer_util-(total_train_util-new_total_util)))
                    print("infer util after: "+str(self.infer_util))
                    print("train util: "+str(self.train_util))
                    print("new_total_util: "+str(new_total_util))
                    print("total_train_util: "+str(total_train_util))
                    print("err_rate: "+str(err_rate))
                    print(self.tasks)
                # assert self.utilization >= -1e-2
                # assert round(100-self.utilization-self.train_util-self.infer_util, 2) <1e-2
                # assert round(100-self.utilization-self.train_util-self.infer_util, 2) >-1e-2
                for task in self.tasks:
                    if task["tagtask"] == 1:
                        # new_util = task["utilization"]/total_infer_util2*(100-new_total_util)
                        new_util = task["utilization"]/total_infer_util2*self.infer_util
                        # if new_util < task["utilization"]:
                        task["speed"] = round((1/0.05)*new_util/task["utilization"],2)
                        # task["speed"] = 1/(0.5*(1/new_util-1/task["control_util"])+1/task["speed"])
                        # else:
                        #     task["speed"] = 1/(self.new_slo*self.slo_factor)
                        if (task["speed"]-1/0.05 < 1e-5) and (task["speed"]-1/0.05 > -1e-5):
                            task["speed"] = 1/0.05
                        if task["speed"]<=0:
                            print("prev_slo: "+str(self.prev_slo))
                            print("overall util: "+str(self.utilization))
                            print("infer util before: "+str(self.infer_util-(total_train_util-new_total_util)))
                            print("infer util after: "+str(self.infer_util))
                            print("train util: "+str(self.train_util))
                            print("new_total_util: "+str(new_total_util))
                            print("total_train_util: "+str(total_train_util))
                            print("infer speed: "+str(task["speed"]))
                            print("sleep: "+str(self.control_sleep))
                            print("new_util: "+str(new_util))
                            print("total_infer_util2: "+str(total_infer_util2))
                            print("total_infer_util: "+str(total_infer_util))
                            print("infer_util: "+str(self.infer_util))
                            print(self.tasks)
                        # assert task["speed"] > -1e-2 #1/0.05
                        task["control_util"] = round(new_util, 5)
                        if new_total_util == 100:
                            print(total_train_util)
                            print("train %d, infer %d"%(self.num_train, self.num_infer))
                            for task in self.tasks:
                                print(task)
                        # assert new_util > -1e-2
                        # if task["speed"] < 0:
                        #     task["speed"] = 0.1*task["control_util"]/total_infer_util
                        #     task["control_util"] = task["control_util"]/total_infer_util
                        # else:
                        #     task["control_util"] = task["control_util"]-train_util_delta*task["control_util"]/total_infer_util
        # recover training utilization after inference is done
        if (self.num_infer == 0) and (self.num_train > 0):
            self.train_util = 0
            for task in self.tasks:
                if task["tagtask"]==0:
                    task["speed"] = 1/0.2
                    task["control_util"] = task["utilization"]
                    self.train_util += task["control_util"]
            self.utilization = 100 - self.train_util
            if self.utilization<0:
                print("\n")
                print(self.tasks)
            # assert self.utilization >= -1e-2
                        
        if (self.num_train == 0) and (self.num_infer > 0):
            self.infer_util = 0
            for task in self.tasks:
                if task["tagtask"]==1:
                    task["speed"] = 1/0.05
                    task["control_util"] = task["utilization"]
                    self.infer_util += task["control_util"]
            self.utilization = 100 - self.infer_util
            # assert self.utilization >= -1e-2

        # if self.num_infer > 0:
        #     self.avglat.append(self.prev_slo)
            
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
        # assert self.utilization >= -1e-2
        task["gpu_ind"] = self.gpuid
        self.tasks.append(task)
        return True
    
    def migrate(self, task):
        self.tasks.remove(task)
        self.memory += task['memory']
        self.utilization += task['utilization']
        assert self.memory <= 32
        # assert self.utilization <= 100+1e-2

