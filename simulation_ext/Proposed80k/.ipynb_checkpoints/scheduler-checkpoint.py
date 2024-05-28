import os
import csv
import pandas as pd
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import utils
from GPU import GPU
from math import ceil
import subprocess
import matplotlib.pyplot as plt
import random
import time
from clock import tic_svc
from controller import Pcontrol_Sleep, Pcontrol_Util
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import copy
###for clearing the screen
def clrscr():
    cls = subprocess.call('cls',shell=True)

###Class Scheduler

class scheduler:
    def __init__(self, alloc_policy=0,\
                 cluster=None,\
                 no_gpus=0,\
                 no_cpus=0,\
                 no_jobs=0,\
                 deadline=0,\
                 control_period=0,\
                 submit_time=[],\
                 duration=[],\
                 jobid=[],\
                 numinst=[],\
                 num_cpu=[],\
                 num_GPU=[],
                task_mem=[],\
                 task_gpu_util=[],\
                 num_inst=[],\
                 start_time=[],\
                 slope_infer=[],\
                slo_factor=1.0):
        self.no_gpus=1
        self.max_no_gpus=no_gpus
        self.deadline=deadline
        self.alloc_policy=alloc_policy
        self.no_cpus=no_cpus
        self.no_jobs=no_jobs
        self.submit_time=submit_time
        self.duration=duration
        self.jobid=jobid
        self.numinst=numinst
        self.num_cpu=num_cpu
        self.num_GPU=num_GPU
        self.task_mem=task_mem
        self.task_gpu_util=task_gpu_util
        self.slope_infer=slope_infer
        self.task_list=[]
        self.gpu_list=[]
        self.gpumemory=[]
        self.num_inst=num_inst
        self.trainingtag=[]
        self.start_time=start_time
        self.used_gpus=[]
        self.control_per = control_period
        self.slo_factor = slo_factor
        
        self.first_job_time = 0
        self.num_day = 0
        self.ddl = []
        self.train_time = []
        self.log_rate = 86400
        self.clock = None
        self.gpu_max_util = 100.0
        
        self.discount = []

###Listing all the GPUs

    def gpu_list(self):
        '''Arranging the GPUs in cluster '''
        no_v100=0 # Train
        no_p100=0 # Infer
        no_k80=0 # Infer
        print("Max No of gpus in datacenter=",self.max_no_gpus)
        # count=0
        # for i in range(0,len(self.num_inst)):
        #     if(self.num_inst[i]>2):
        #         count=count+1
        #         self.trainingtag.append(i)
        # print("Training task", count)
        for i in range(0,self.max_no_gpus):
            if i%2==0:
                no_v100=no_v100+1
                self.gpu_list.append(1)
            elif i%2!=0 and i%5==0:
                no_p100=no_p100+1
                self.gpu_list.append(2)
            elif i%2!=0 and i%7==0:
                no_k80=no_k80+1
                self.gpu_list.append(3)
            elif i%2!=0 and i%3==0:
                no_p100=no_p100+1
                self.gpu_list.append(2)
            elif i%2!=0 and i%11==0:
                no_k80=no_k80+1
                self.gpu_list.append(3)
            elif i%2!=0 and i%13!=0:
                no_p100=no_p100+1
                self.gpu_list.append(2)
        #print(len(self.gpu_list))
        x=self.max_no_gpus-len(self.gpu_list)
        #print(x)
        if(x!=0):
            for i in range(0,x):
                self.gpu_list.append(1)
                no_v100=no_v100+1
        print("No of K80 =",no_k80)
        print("No of V100 =",no_v100)
        print("No of P10 =",no_p100)

    ###Assigning memory for the GPU

    def gpumemoryassignment(self):
        x=len(self.gpu_list)
        if(x-self.max_no_gpus==0):
            for i in range(0,self.max_no_gpus):
                if(self.gpu_list[i]==1):
                    self.gpumemory.append(32) #32
                elif(self.gpu_list[i]==2):
                    self.gpumemory.append(32) #24
                else:
                    self.gpumemory.append(32) #16
        else:
            print("Error in datacenter gpu allocation")

    def gpuutilassignment(self):
        count=0
        self.tagtask=[]
        self.inferencecount_trace=0
    
        time_elapsed = self.task_list[-1]['start_time']-self.task_list[0]['start_time']
        threshold_u = 20
        for i in range(0,len(self.task_gpu_util)):
            time_to_now = (self.task_list[i]['start_time']-self.task_list[0]['start_time'])/3600
            # if((time_to_now<24) or (time_to_now>=48)):
            #     threshold_t = 50000 # 3000
            #     threshold_u = 20
            # else:
            threshold_t = 50000 # 59000
            threshold_u = 20 # 26
            if(self.task_list[i]['duration']>threshold_t): #60
            # if(self.duration[i]>20000):
                count=count+1
                self.tagtask.append(0) # training
            else:
                if(self.task_list[i]['utilization']>threshold_u):
                    self.tagtask.append(0) # training
                else:
                    self.tagtask.append(1) # inference
                # print("Job Tag: ")
                # print(self.tagtask)
                self.inferencecount_trace=self.inferencecount_trace+1
            
        print(f"Training job = {count}")

        taskleng=len(self.jobid)

        for i in range(taskleng):
            if(i==taskleng-1):
                self.finaljob=(self.jobid[i])

      # print(self.tagtask)



     

    '''
    Default scheduler without deadline misses
    '''


    '''
    Scheduler with Job misses
    '''

    def missline(self,job_ind,clock): # schedule missed jobs
        # print(f"IN MISSLINE TASK {kk}\n")
        rnd2_schedule=False
        rnd2_gpu = []
        for k in range(self.no_gpus):
            # tager,gpuno, task: dict, start_time
            # print("k=%d,kk=%d, start_time %d, task_list %d, objs %d"%(k,kk,len(self.start_time), len(self.task_list),len(self.objs)))
            x=self.objs[k].assign_task(k,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
            if(x==True):
                # print("M:Time %.2f Start time %ds GPU %d Task %d Tag %d ReUtil = %.3f per ReMem %.3f GB"%(clock.get_cur_time(),self.start_time[kk],k,kk,self.tagtask[kk],self.objs[k].utilization,self.objs[k].memory))
                # if self.objs[k].utilization<0:
                #     print(self.objs[k].tasks)
                #     print(self.task_list[job_ind])
                #     util=0
                #     for t in self.objs[k].tasks:
                #         util = util+t["utilization"]
                #     print("total util %.1f"%(util))
                #     exit()
                # assert self.objs[k].utilization<0
                # self.initialschedulegpu.append(k)
                return True
            elif(x==-1):
                rnd2_schedule=True
                if len(rnd2_gpu)==0:
                    rnd2_gpu.append(k)
        if rnd2_schedule==True:
            gpu_ind = rnd2_gpu[0]
            x=self.objs[gpu_ind].assign_task2(gpu_ind,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
            assert x==True
            return True
                
        return False

    def gpu_count(self):
        local_gpu_count=0
        for k in range(self.no_gpus):
           self.objs[k].gpu_count()
           local_gpu_count=local_gpu_count+self.objs[k].gpucount
        self.gpu_datacenter.append(local_gpu_count)


#*************old tic job*******
    def tic_clock_job(self, clock):
        clock.tic_job() # increase simulated time #
        # if (clock.get_cur_time()>=494) and (clock.get_cur_time()<=560):
        #     print("scheduler time: "+str(clock.get_cur_time()))
        finished_jobs=0
        sum_train = 0
        sum_infer = 0
        start_t = time.time()
        for qqk in range(self.no_gpus):
            # executor.submit(self.process_gpu,clock,self.objs[qqk])
            # print("Loop GPUs!")
            # print("\ncurrent time %d"%clock.get_cur_time())
            finished_jobs = finished_jobs+self.objs[qqk].taskrun
            sum_train = sum_train+self.objs[qqk].num_train
            sum_infer = sum_infer+self.objs[qqk].num_infer
            if round(clock.get_cur_time(),0) % self.control_per == 0:
                # print("Time: %d, inner loop"%((clock.get_cur_time()-self.task_list[0]['start_time'])/self.control_per))         
                # Consolidation
                # train_tasks = []
                # for ind_gpu in range(self.no_gpus):
                #     if len(self.objs[ind_gpu].tasks) > 0:
                #         if self.objs[ind_gpu].num_infer == 0:
                #             cloned_tasks = self.objs[ind_gpu].tasks.copy()
                #             train_tasks = train_tasks+cloned_tasks
                #         if self.objs[ind_gpu].num_train == 0:
                #             if len(train_tasks) > 0:
                #                 min_train = min(train_tasks, key=lambda x: (x['utilization'], x['memory']))
                #                 if self.objs[ind_gpu].admit_migrate(min_train):
                #                     self.objs[min_train['gpu_ind']].migrate(min_train)
                #                     train_tasks.remove(min_train)
                        
                if round(clock.get_cur_time(),0) % (self.control_per*25) == 0:
                    enable_outer_loop = True
                    # print("Time: %d, outer loop"%((clock.get_cur_time()-self.task_list[0]['start_time'])/self.control_per/25))
                else:
                    enable_outer_loop = False
                self.objs[qqk].invoke_control(enable_outer_loop)
                # None
                # print("Invoke Control")
            self.objs[qqk].remove_finished_tasks()
            # task_cnt.append(len(self.objs[qqk].tasks))

        if round(clock.get_cur_time(),0) % 1 == 0:
            used_gpu_num = 0
            used_gpu_train = 0
            used_gpu_infer = 0
            used_gpu_colo = 0
            if round(clock.get_cur_time(),0) % 1 == 0:
                for ind_gpu in range(self.no_gpus):
                    if len(self.objs[ind_gpu].tasks) > 0:
                        used_gpu_num = used_gpu_num+1
                        if self.objs[ind_gpu].num_infer == 0:
                            used_gpu_train = used_gpu_train+1
                        elif self.objs[ind_gpu].num_train == 0:
                            used_gpu_infer = used_gpu_infer+1
                        else:
                            used_gpu_colo = used_gpu_colo+1
                        # Debug
                        # if (clock.get_cur_time()-self.task_list[0]['start_time']<=169200):
                        #     if self.objs[ind_gpu].new_slo < 0.05:
                        #         for t in self.objs[ind_gpu].tasks:
                        #             if t["tagtask"] == 1:
                        #                 if t["start_time"]<t["start"]:
                        #                     dist = (t["start_time"]+t["duration"]-t["start"])/t["duration"]
                        #                     self.discount.append(dist)
                                            
                        
            self.used_gpus.append(used_gpu_num)
        end_t = time.time()

        # if round(clock.get_cur_time()-self.task_list[0]['start_time']) == 79200:
        #     print("Log")
        #     for ind_gpu in range(self.no_gpus):
        #         if len(self.objs[ind_gpu].tasks) > 0:
        #             if self.objs[ind_gpu].num_infer == 0:
        #                 with open('train.txt', 'a') as f:
        #                     for t in self.objs[ind_gpu].tasks:
        #                         f.write('GPU %d, Util %.2f, Mem %.2f, Dur %d, Tag %d'%(ind_gpu,t['utilization'],t['memory'],t['duration'],t['tagtask']))
        #     exit()
                
        if round(clock.get_cur_time()-self.first_job_time,0) % self.log_rate == 0:
            self.num_day = self.num_day+1
            self.dump_data(self.num_day)
        if round(clock.get_cur_time(),0) % self.log_rate==0:
            print("***STime %.2fs, Loop %.2fms, GPUs(T/I/C/A/CA) %d/%d/%d/%d/%d, Finished %d/%d, Train %d, Infer %d***"%(clock.get_cur_time()-self.task_list[0]['start_time'],(end_t-start_t)*1000,used_gpu_train,used_gpu_infer,used_gpu_colo,used_gpu_num,self.no_gpus,finished_jobs,self.no_jobs, sum_train, sum_infer))


#***************************Original*********************************
    def dispatch_job_to_gpu(self, schedule_job_ind):
        success_job_ind = []
        infer_only = []
        train_only = []
        infer_train = []
        empty = []
        for gpu_ind in range(self.no_gpus):
            if (self.objs[gpu_ind].num_infer > 0) and (self.objs[gpu_ind].num_train == 0):
                infer_only.append({"ind": gpu_ind, "mem": self.objs[gpu_ind].memory, "util": self.objs[gpu_ind].utilization})
            elif (self.objs[gpu_ind].num_infer == 0) and (self.objs[gpu_ind].num_train > 0):
                train_only.append({"ind": gpu_ind, "mem": self.objs[gpu_ind].memory, "util": self.objs[gpu_ind].utilization})
            elif (self.objs[gpu_ind].num_infer > 0) and (self.objs[gpu_ind].num_train > 0):
                infer_train.append({"ind": gpu_ind, "mem": self.objs[gpu_ind].memory, "util": self.objs[gpu_ind].utilization})
            else:
                empty.append(gpu_ind)
            # infer_only = sorted(infer_only, key=lambda d: (-d['mem'], -d['util']))
            # train_only = sorted(train_only, key=lambda d: (-d['mem'], -d['util']))
            # infer_train = sorted(infer_train, key=lambda d: (-d['mem'], -d['util']))
            
        for job_ind in schedule_job_ind:    
            schedule=False
            if self.task_list[job_ind]['tagtask'] == 0: # train
                # 1st: has inference only and has largest avaialble mem and util
                if len(infer_only) > 0:
                    for i, g in enumerate(infer_only):
                        schedule=self.objs[g["ind"]].assign_task_train(g["ind"],self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            infer_only.pop(i)
                            infer_train.append({"ind": g["ind"], "mem": self.objs[g["ind"]].memory, "util": self.objs[g["ind"]].utilization})
                            # infer_train = sorted(infer_train, key=lambda d: (-d['mem'], -d['util']))
                            break
                # 2nd: has inference and training and has largest available mem and util
                if (schedule==False) and (len(infer_train) > 0):
                    for g in infer_train:
                        schedule=self.objs[g["ind"]].assign_task_train(g["ind"],self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            # infer_train = sorted(infer_train, key=lambda d: (-d['mem'], -d['util']))
                            break
                # 3rd: has training only and has largest available mem and util
                if (schedule==False) and (len(train_only) > 0):
                    for i, g in enumerate(train_only):
                        schedule=self.objs[g["ind"]].assign_task_train(g["ind"],self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            # train_only = sorted(train_only, key=lambda d: (-d['mem'], -d['util']))
                            break                
                # 4th: empty GPU
                if (schedule==False) and (len(empty) > 0):
                    for i in empty:
                        schedule=self.objs[i].assign_task_train(i,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            empty.remove(i)
                            train_only.append({"ind": i, "mem": self.objs[i].memory, "util": self.objs[i].utilization})
                            # train_only = sorted(train_only, key=lambda d: (-d['mem'], -d['util']))
                            break 
                            
            if self.task_list[job_ind]['tagtask'] == 1: # infer
                # 1st: has training only and has largest available mem and util
                if len(train_only) > 0:
                    for i, g in enumerate(train_only):
                        schedule=self.objs[g["ind"]].assign_task_infer(g["ind"],self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            train_only.pop(i)
                            infer_train.append({"ind": g["ind"], "mem": self.objs[g["ind"]].memory, "util": self.objs[g["ind"]].utilization})
                            # infer_train = sorted(infer_train, key=lambda d: (-d['mem'], -d['util']))
                            break
                # 2nd: has training and inference and has largest available mem and util
                if (schedule==False) and (len(infer_train) > 0):
                    for g in infer_train:
                        schedule=self.objs[g["ind"]].assign_task_infer(g["ind"],self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            # infer_train = sorted(infer_train, key=lambda d: (-d['mem'], -d['util']))
                            break
                # 3rd: has inference only and has largest available mem and util
                if (schedule==False) and (len(infer_only) > 0):
                    for i, g in enumerate(infer_only):
                        schedule=self.objs[g["ind"]].assign_task_infer(g["ind"],self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            # infer_only = sorted(infer_only, key=lambda d: (-d['mem'], -d['util']))
                            break      
                # 4th: empty GPU
                if (schedule==False) and (len(empty) > 0):
                    for i in empty:
                        schedule=self.objs[i].assign_task_infer(i,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        if schedule:
                            empty.remove(i)
                            infer_only.append({"ind": i, "mem": self.objs[i].memory, "util": self.objs[i].utilization})
                            # infer_only = sorted(infer_only, key=lambda d: (-d['mem'], -d['util']))
                            break 
                    
#             rnd3_schedule=False # for rnd2_gpu3
#             rnd2_gpu0 = [] # at least one trainng or inference
#             rnd2_gpu = [] # has at least one job but no exclusive
#             rnd2_gpu2 = [] # empty GPUs
#             rnd2_gpu3 = [] # has only one training and uses almost all util
#             # if round(clock.get_cur_time(),0) == 75600:
#             #     exit()
#             for gpu_ind in range(self.no_gpus):
#                 #print(k)
#                 x=self.objs[gpu_ind].assign_task(gpu_ind,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
#                 if(x==True):
#                     rnd2_gpu0.append({"ind": gpu_ind, "num_infer": self.objs[gpu_ind].num_infer})
#                     # self.initialschedulegpu.append(gpu_ind)
#                     # schedule=True
#                     # break
#                 elif(x==-1):
#                     # rnd2_schedule=True
#                     # if (len(rnd2_gpu)==0) and (self.objs[gpu_ind].num_infer+self.objs[gpu_ind].num_train>0):
#                     if self.objs[gpu_ind].num_infer+self.objs[gpu_ind].num_train>0:
#                         rnd2_gpu.append({"ind": gpu_ind, "num_jobs": self.objs[gpu_ind].num_infer+self.objs[gpu_ind].num_train})
#                     else:
#                         rnd2_gpu2.append(gpu_ind) # empty GPU
#                 elif(x==-2):
#                     rnd2_gpu3.append(gpu_ind)
#                 else:
#                     schedule=False

#             # if (rnd2_schedule==True) and (schedule==False):
#             if (len(rnd2_gpu0)>0) or (len(rnd2_gpu)>0) or (len(rnd2_gpu2)>0) or (len(rnd2_gpu3)>0):
#                 if len(rnd2_gpu0) > 0:
#                     rnd2_gpu0=sorted(rnd2_gpu0, key=lambda d: (d['num_infer']))
#                     gpu_ind = rnd2_gpu0[0]['ind']
#                 elif len(rnd2_gpu) > 0:
#                     rnd2_gpu=sorted(rnd2_gpu, key=lambda d: (d['num_jobs']))
#                     gpu_ind = rnd2_gpu[0]['ind']
#                 elif len(rnd2_gpu2) > 0:
#                     gpu_ind = rnd2_gpu2[0]
#                 else:
#                     gpu_ind = rnd2_gpu3[0]
#                     rnd3_schedule = True
#                 if rnd3_schedule:
#                     x=self.objs[gpu_ind].assign_task3(gpu_ind,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
#                     # print("Schedule 1v2 GPU %d"%gpu_ind)
#                 else:
#                     x=self.objs[gpu_ind].assign_task2(gpu_ind,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
#                 assert x==True
#                 schedule = True
#             # elif schedule==False:
            if schedule==False:
                # print("Critical Error: No GPUs Available!")
                self.no_gpus = self.no_gpus+1
                if self.no_gpus <= self.max_no_gpus:
                    self.objs.append(GPU(memory=self.gpumemory[self.no_gpus-1],\
                                         utilization=self.gpu_max_util,\
                                         dram=self.gpumemory[self.no_gpus-1],
                                         clock=self.clock,\
                                         ddl = self.ddl,\
                                         train_time = self.train_time,\
                                         slo_factor=self.slo_factor))
                    if self.task_list[job_ind]['tagtask'] == 0: # train
                        x=self.objs[-1].assign_task_train(self.no_gpus-1,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                    else: # infer
                        x=self.objs[-1].assign_task_infer(self.no_gpus-1,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                    assert x==True
                    schedule = True
                else:
                    self.no_gpus = self.no_gpus-1
                    schedule = False
            # Log successful job ind
            if schedule == True:
                success_job_ind.append(job_ind)
        return success_job_ind
#***************************Original*********************************
    def deadlinemissscheduler(self,clock):
        self.plotinggpuassignment=[]
        missed_jobs=[]
        #import time
        start_time=0
        start_time2=0
        interimtime=0
        infersum=0
        intercounter=0
        self.initialschedulegpu=[]
        self.finalschedulegpu=[]
        self.gpu_datacenter=[]
        self.schedulingtime=[]
        self.timeschedulegpu=[]
        
        pass_job_ind = np.arange(0, self.no_jobs).tolist()
        while len(pass_job_ind)>0:
            self.tic_clock_job(clock)
            schedule_job_ind = []
            for job_ind in pass_job_ind:
                if clock.get_cur_time() >= self.task_list[job_ind]['start_time']:
                    schedule_job_ind.append(job_ind)
                else:
                    break

            success_job_ind = self.dispatch_job_to_gpu(schedule_job_ind)
            for job_ind in success_job_ind:
                pass_job_ind.remove(job_ind)
                
            # check missed jobs after every 100 jobs
#             if((i%100==0 and len(missed_jobs)>=1) or (i==self.no_jobs-1 and len(missed_jobs)>=1)):
#                 #print(scheduled)
#                 cnt = 0
#                 while(len(missed_jobs) > 0):
#                     self.gpu_count()
#                     while(True):

#                         ch=self.missline(missed_jobs[0],clock)
#                         if ch:
#                             break
#                         # print("\n Missed Jobs Current time %d"%(clock.get_cur_time()))
#                         # task_cnt = []
#                         # with ThreadPoolExecutor(max_workers=1000) as executor:
#                         sum=0
#                         sum_train = 0
#                         sum_infer = 0
#                         start_t = time.time()
#                         for qqk in range (self.no_gpus):
#                                 # executor.submit(self.process_gpu,clock,self.objs[qqk])
#                             # print("Loop GPUs!")
#                             # print("\ncurrent time %d"%clock.get_cur_time())
#                             sum=sum+self.objs[qqk].taskrun
#                             sum_train = sum_train+self.objs[qqk].num_train
#                             sum_infer = sum_infer+self.objs[qqk].num_infer
#                             if round(clock.get_cur_time(),0) % self.control_per == 0:
#                                 self.objs[qqk].invoke_control()
#                                 # None
#                                 # print("Invoke Control")
#                             self.objs[qqk].remove_finished_tasks()
#                             # task_cnt.append(len(self.objs[qqk].tasks))

#                         if round(clock.get_cur_time(),0) % 1 == 0:
#                             used_gpu_num = 0
#                             for ind_gpu in range(self.no_gpus):
#                                 if len(self.objs[ind_gpu].tasks) > 0:
#                                     used_gpu_num = used_gpu_num+1
#                             self.used_gpus.append(used_gpu_num)  
#                         end_t = time.time()
#                         clock.tic_job() # increase simulated time
#                         if round(clock.get_cur_time()-self.first_job_time,0) % self.log_rate == 0:
#                             self.num_day = self.num_day+1
#                             self.dump_data(self.num_day)
#                         if round(clock.get_cur_time(),0) % 60==0:
#                             print("***MTime %.2fs, Loop %.2fs, GPUs %d/%d, Finished %d/%d, Train %d, Infer %d, MISS %d***"%(clock.get_cur_time()-self.start_time[0],end_t-start_t,used_gpu_num,self.no_gpus,sum,self.no_jobs, sum_train, sum_infer, len(missed_jobs)))
#                             # if len(missed_jobs)==54:
#                             #     for index in missed_jobs:
#                             #         print(self.task_list[index])
#                         # if round(clock.get_cur_time(),1) == 20.0:
#                         #     print("\nYes %f"%clock.get_cur_time())
#                         # print(f"Miss Remain Jobs {self.no_jobs-i-1} Num of Tasks {task_cnt}")
#                         # if len(missed_jobs)==1:
#                         #     print(self.task_list[missed_jobs[0]])

#                     missed_jobs.pop(0)

            # if(infersum==self.inferencecount_trace): # reach the total num of inferences in the trace
            #     print(f"Inter Execution time ={clock.get_cur_time()} seconds")
            #     interimtime=(interimtime+clock.get_cur_time())
            #     infersum=infersum+10
            #     start_time2=clock.get_cur_time()
            #     intercounter=intercounter+1

            # if(i==self.no_jobs-1):
        print("Waiting for tasks to complete")
        scheduled_flag=False
        while(scheduled_flag==False):
            clock.tic_job() # increase simulated time
            finished_jobs=0

            # print("\n After Missing Jobs Current time %d"%(clock.get_cur_time()))
            start_t = time.time()
            used_gpu_num = 0
            used_gpu_train = 0
            used_gpu_infer = 0
            used_gpu_colo = 0
            if round(clock.get_cur_time(),0) % 1 == 0:
                for ind_gpu in range(self.no_gpus):
                    if len(self.objs[ind_gpu].tasks) > 0:
                        used_gpu_num = used_gpu_num+1
                        if self.objs[ind_gpu].num_infer == 0:
                            used_gpu_train = used_gpu_train+1
                        elif self.objs[ind_gpu].num_train == 0:
                            used_gpu_infer = used_gpu_infer+1
                        else:
                            used_gpu_colo = used_gpu_colo+1
                        # debug
                                
                                    
                        # if (clock.get_cur_time()-self.task_list[0]['start_time']<=169200):
                        #     if self.objs[ind_gpu].new_slo < 0.05:
                        #         for t in self.objs[ind_gpu].tasks:
                        #             if t["tagtask"] == 1:
                        #                 if t["start_time"]<t["start"]:
                        #                     dist = (t["start_time"]+t["duration"]-t["start"])/t["duration"]
                        #                     self.discount.append(dist)
                        # else:
                        #     break
                self.used_gpus.append(used_gpu_num)  

            # task_cnt = []
            # with ThreadPoolExecutor(max_workers=1000) as executor:
            #     for qqk in range (self.no_gpus):
            #         executor.submit(self.process_gpu,clock,self.objs[qqk])
            sum_train = 0
            sum_infer = 0
            for qqk in range (self.no_gpus):
                # self.objs[qqk].remove_finished_tasks()
                # task_cnt.append(len(self.objs[qqk].tasks))
                # print("Loop GPUs!")
                # print("\ncurrent time %d"%clock.get_cur_time())
                if round(clock.get_cur_time(),0) % self.control_per == 0:
                    # print("Time: %d, inner loop"%((clock.get_cur_time()-self.task_list[0]['start_time'])/self.control_per))                        
                    if round(clock.get_cur_time(),0) % (self.control_per*25) == 0:
                        enable_outer_loop = True
                        # print("Time: %d, outer loop"%((clock.get_cur_time()-self.task_list[0]['start_time'])/self.control_per/25))
                    else:
                        enable_outer_loop = False
                    self.objs[qqk].invoke_control(enable_outer_loop)
                    # None
                    # print("Invoke Control")

                self.objs[qqk].remove_finished_tasks()
                # record the num of used gpus
                sum_train = sum_train+self.objs[qqk].num_train
                sum_infer = sum_infer+self.objs[qqk].num_infer
                finished_jobs = finished_jobs+self.objs[qqk].taskrun
                # infersum=infersum+self.objs[qqk].inferenceocount_sim
                # if(infersum==self.inferencecount_trace):
                #     print(f"Inter Execution time ={clock.get_cur_time()} seconds")
                #     interimtime = (interimtime + clock.get_cur_time())
                #     # infersum=infersum+10
                #     intercounter = intercounter + 1
                #     start_time2 = clock.get_cur_time()

            if(finished_jobs==self.no_jobs):
                scheduled_flag=True
                self.num_day = self.num_day+1
                self.dump_data(self.num_day)
                # print(f"Remain Jobs {self.no_jobs-i-1} Num of Tasks {task_cnt}")
                break
            else:
                scheduled_flag=False
            end_t = time.time()
            
            if round(clock.get_cur_time()-self.first_job_time,0) % self.log_rate == 0:
                self.num_day = self.num_day+1
                self.dump_data(self.num_day)
            if round(clock.get_cur_time(),0) % self.log_rate==0:
                print("***Time %.2fs, Loop %.2fms, GPUs(T/I/C/A/CA) %d/%d/%d/%d/%d, Finished %d/%d, Train %d, Infer %d***"%(clock.get_cur_time()-self.task_list[0]['start_time'],(end_t-start_t)*1000,used_gpu_train,used_gpu_infer,used_gpu_colo,used_gpu_num,self.no_gpus,finished_jobs,self.no_jobs, sum_train, sum_infer))
                            # if round(clock.get_cur_time(),1) == 20.0:
                            #     print("\nYes %f"%clock.get_cur_time())
                            # print(f"Remain Jobs {self.no_jobs-i-1} Num of Tasks {task_cnt}")

                # clock.tic_job() # increase simulated time
        # temp=[]
        # empt=[]
        # for i in range(self.no_gpus):
        #     if self.objs[i].latency:
        #         temp.append(self.objs[i].latency)
        # count=0
        # for x in range(0,len(temp)):
        #     for y in temp[x]:
        #         count=count+1
        #         empt.append({'Latency':y['Latency'],'Slno':y['Jobid'],'Utilization':y['Utilization']})
                

        # newlist = sorted(empt, key=lambda d: d['Slno'])
        # Slno: sorted job id based after simulation.
        #print(newlist)
        #print(newlist[1]['Utilization'])

        # count=0
        # temp1=0
        # delta=0.6

        # print("Finsihed iteration 1")
        #scheduled.clear()

        # for yyy in range (self.no_gpus):
        #     self.objs[yyy].taskrun=0
        #     self.objs[yyy].latency.clear()

        # final_time=(clock.get_cur_time()-start_time2)
        # print(f"final time ={final_time} curent time {clock.get_cur_time()}")
        # self.cost=0
        # self.cost=self.cost+((self.no_gpus/2)*final_time) +(interimtime*self.no_gpus)
        # print(f"Final Cost ={self.cost}")
            
    def dump_data(self,num_day):
        with open('data/used_gpus/used_gpus_slo%d_day%d'%(self.slo_factor*100,num_day), 'w') as fp:
            for item in self.used_gpus:
                fp.write("%s\n" % item)
        self.used_gpus.clear()
        
        if len(self.ddl) > 0:
            miss_ddl_rate = 1-np.sum(self.ddl)/len(self.ddl)
        else:
            miss_ddl_rate = 0
        with open('data/miss_ddl/miss_ddl_slo%d_day%d'%(self.slo_factor*100,num_day), 'w') as fp:
            fp.write("%s\n" % miss_ddl_rate)
        self.ddl.clear()
        print("SLO Violation %f"%(miss_ddl_rate*100))
        
        with open('data/train_time/train_time_slo%d_day%d'%(self.slo_factor*100,num_day), 'w') as fp:
            for item in self.train_time:
                fp.write("%s\n" % item)
        self.train_time.clear()

        with open('data/dist/discount_day%d.txt'%(num_day), 'w') as fp:
            for item in self.discount:
                fp.write("%s\n" % item)
        self.discount.clear()
                
    
    '''
    The default function from the main function!!!!!!!!! comes here after scheduler
    '''
    def schedule(self):
        # self.utilization=150.0 # change later
        # time=0
        gpuspecs =	{
        "V100":32,
        "K80": 24,
        "P100":16,
        }   
        for i in range(len(self.jobid)):
            self.task_list.append({'Jobid':self.jobid[i],\
                               'memory':self.task_mem[i],\
                               'utilization':round(self.task_gpu_util[i], 5),\
                               'duration':self.duration[i],\
                               'slno':i,\
                               'start_time':self.start_time[i],\
                               'meet_ddl':True,\
                               'delay':False})
            # print("\njob tags:")
            # print(self.tagtask[i])

        self.task_list = sorted(self.task_list, key=lambda d: (d['start_time'], -d['utilization'])) # sort ascending order based on start time and utilization

        self.first_job_time = self.task_list[0]['start_time']
        for i in range(len(self.task_list)):
            self.task_list[i]['start_time'] = self.task_list[i]['start_time']-self.first_job_time
        self.first_job_time=0
        scheduler.gpu_list(self)
        scheduler.gpumemoryassignment(self)
        self.gpuutilassignment()
        print("Total no of GPUS =",self.no_gpus)
        print("Length of GPU MEMORY=",len(self.gpumemory))
        # print("Final Job =",self.finaljob)
        for i in range(len(self.jobid)):
            self.task_list[i]['tagtask'] = self.tagtask[i]

        count=0
        self.initiallatency=[]
        self.job_altered_latency=[]

        temp1=0
        # delta=0.6 # inintial latency increase rate
     #######################################################################       
        # hour_cnt = 1
        # thresh_t = 50000
        # thresh_u = 20
        # train_cnt = 0
        # infer_cnt = 0
        # train_cnt_list = []
        # infer_cnt_list = []
        # for i,task in enumerate(self.task_list):
        #     if task['start_time']/3600/24 <= hour_cnt:
        #         if (task['utilization'] > thresh_t):
        #             train_cnt=train_cnt+1
        #         else:
        #             if task['utilization'] > thresh_u:
        #                 train_cnt=train_cnt+1
        #             else:
        #                 infer_cnt=infer_cnt+1
        #     else:
        #         train_cnt_list.append(train_cnt)
        #         infer_cnt_list.append(infer_cnt)
        #         train_cnt=0
        #         infer_cnt=0
        #         if (task['utilization'] <= thresh_t) and (task['utilization'] <= thresh_u): 
        #             infer_cnt=infer_cnt+1
        #         else:
        #             train_cnt=train_cnt+1
        #         hour_cnt = hour_cnt+1
        # np.save("train_cnt.npy", np.array(train_cnt_list))
        # np.save("infer_cnt.npy", np.array(infer_cnt_list))
        # exit()
        ###################################################################

        self.clock = tic_svc(cur_time=round(self.task_list[0]['start_time']), delta=10.0) # create universal clock
        # print("start time %f, end time %f"%(self.task_list[0]['start_time'],self.task_list[-1]['start_time']))
        controller_sleep = 0
        controller_util = 0
        
        # initialize one GPU first and increase on demand
        self.objs = [GPU(memory=self.gpumemory[0],\
                         utilization=self.gpu_max_util,\
                         dram=self.gpumemory[0],\
                         clock=self.clock,\
                         ddl = self.ddl,\
                         train_time = self.train_time,\
                         slo_factor=self.slo_factor)]
        #scheduler.allocate(self,self.task_list,objs)
        print("No of jobs =",len(self.jobid))
        print("\n")
        #clrscr()
        latency=[]
        temp=[]
        tempid=[]
        import time
        if(self.deadline==0):
            print("regular scheduler is not enabled. I quit!")
            exit();

        else:
            print("Running scheduler with deadline misses")
            s_time = time.time()
            ss_time = self.clock.get_cur_time()
            self.deadlinemissscheduler(self.clock)
            e_time = time.time()
            ee_time = self.clock.get_cur_time()
            
            print("Finished all jobs! Real Exe. Time %f s, Simu Exe. Time %.2f s"%(e_time-s_time, ee_time-ss_time))
            for i in range(self.no_gpus):
                if self.objs[i].latency:
                    #print(self.objs[i].latency)
                    temp.append(self.objs[i].latency)
                    
            #print(temp)
            #print(len(temp))
            count=0
            empt=[]

            #for x in range(0,len(temp)):
             #   for y in temp[x]:
              #      print(y)
               #     count=count+1
                #    #print(temp[x][y])

            #print(count)
            
            for x in range(0,len(temp)):
                for y in temp[x]:
                    count=count+1
                    empt.append({'Latency':y['Latency'],'Slno':y['Jobid']})

            newlist = sorted(empt, key=lambda d: d['Slno'])
            #print((newlist))
            
            total_infer = 0
            miss_ddl = 0
            for task in self.task_list:
                if task["tagtask"] == 1:
                    total_infer = total_infer+1
                    if task["meet_ddl"] == False:
                        miss_ddl = miss_ddl+1
            # with open('data/missed_ddl_jobs_%d'%(self.slo_factor*100), 'w') as fp:
            #     fp.write("%s\n" % (miss_ddl/total_infer*100))
            print('Missed ddl jobs written %.2f'%(miss_ddl/total_infer*100))         
            
            # for i,gpu in enumerate(self.objs):
            #     with open('data/avg_lat/avg_lat_gpu%d'%(i+1), 'w+') as fp:
            #         gpu_avglat = np.array(gpu.avglat)
            #         for item in gpu_avglat:
            #             fp.write("%s\n" % item)
            print("Length of inference =",len(newlist))
        
            # for i in range(len(newlist)):
            #     latency.append(newlist[i]['Latency'])
            #     #tempid.append(newlist[i]['Slno']) 
            #     tempid.append(i) 
                #print(newlist[i]['Slno'])
                #print(newlist[i]['Latency'])
            # with open('data/used_gpus_%d'%(self.slo_factor*100), 'w') as fp:
            #     for item in self.used_gpus:
            #         fp.write("%s\n" % item)
            # print("Used GPU written, len %d"%(len(self.used_gpus)))
            
            # with open('data/latency', 'w') as fp:
            #     for item in latency:
            #         fp.write("%s\n" % item)
            # print('Latency written')


                
            # with open('data/tempid', 'w') as fp:
            #     for item in tempid:
            #         fp.write("%s\n" % tempid)
            # print('Job id written')



           # with open('data/gpudatacenter', 'w') as fp:
            #    for item in self.gpu_datacenter:
             #       fp.write("%s\n" % item['Time'])
            #print('timestamp written')

            # with open('data/gpudatatimestamp', 'w') as fp:
            #     for item in self.gpu_datacenter:
            #         fp.write("%s\n" % item)
            # print('gpudatacenter written')

            # with open('data/schedulingtimestamp', 'w') as fp:
            #     for item in self.timeschedulegpu:
            #         fp.write("%s\n" % item['Timestamp'])
            # print('timescheduling written')

            # with open('data/schedulingtimejob', 'w') as fp:
            #     for item in self.timeschedulegpu:
            #         fp.write("%s\n" % item['Jobid'])
            # print('Job written')



