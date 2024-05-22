import os
import csv
import pandas as pd
import multiprocessing
import threading
import numpy as np
import utils
from GPU import GPU
from math import ceil
import subprocess
import matplotlib.pyplot as plt
import random
import time
from clock import tic_svc
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
                 submit_time=[],\
                 duration=[],\
                 jobid=[],\
                 numinst=[],\
                 num_cpu=[],\
                 num_GPU=[],
                task_mem=[],\
                 task_gpu_util=[],\
                 num_inst=[],\
                 slope_infer=[],\
                start_time=[],\
                slo_factor=1.0):
        self.max_no_gpus=no_gpus
        self.max_no_gpu_infer=int(self.max_no_gpus/6*5)
        self.max_no_gpu_train=int(self.max_no_gpus/6)
        self.no_gpu_train=1
        self.no_gpu_infer=1
        # self.no_gpu_infer=1
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
        self.used_gpus_train=[]
        self.used_gpus_infer=[]
        
        self.first_job_time = 0
        self.num_day = 0
        self.ddl = []
        self.train_time = []
        self.log_rate = 86400
        self.clock = None
        self.gpu_max_util = 100.0
        self.obj_train = None
        self.obj_infer = None
        self.slo_factor = slo_factor

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
            # time_to_now = (self.task_list[i]['start_time']-self.task_list[0]['start_time'])/3600
            # if((time_to_now<24) or (time_to_now>=48)):
            #     threshold_t = 50000 # 3000
            #     threshold_u = 20
            # else:
            #     threshold_t = 50000 # 59000
            #     threshold_u = 60 # 26
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
    Scheduler with Job misses
    '''

    def missline(self,kk): # schedule missed jobs
        # print(f"IN MISSLINE TASK {kk}\n")
      
        for k in range(self.no_gpus):
            x=self.objs[k].assign_task(k,self.task_list[kk],self.start_time[kk])
            if(x==True):
                # print("M:Time %.2f Start time %ds GPU %d Task %d Tag %d ReUtil = %.3f per ReMem %.3f GB"%(clock.get_cur_time(),self.start_time[kk],k,kk,self.tagtask[kk],self.objs[k].utilization,self.objs[k].memory))
                # print(f"Start time {self.start_time[k]}s GPU {k} assigned for task {kk} total utilization for the GPU = {self.objs[k].utilization} % pending memory for the gpu {self.objs[k].memory} GB")
                self.initialschedulegpu.append(k)
                return True
        return False

    def gpu_count(self):
        local_gpu_count=0
        for k in range(self.no_gpus):
           self.objs[k].gpu_count()
           local_gpu_count=local_gpu_count+self.objs[k].gpucount
        self.gpu_datacenter.append(local_gpu_count)

    def tic_clock_job(self, clock):
        clock.tic_job() # increase simulated time #
        finished_jobs=0
        sum_train = 0
        sum_infer = 0
        start_t = time.time()
        for qqk in range(self.no_gpu_train):
            self.obj_train[qqk].remove_finished_tasks()
            finished_jobs=finished_jobs+self.obj_train[qqk].taskrun
            sum_train = sum_train+self.obj_train[qqk].num_train

            # task_cnt.append(len(self.obj_train[qqk].tasks))
        for qqk in range(self.no_gpu_infer):
            self.obj_infer[qqk].remove_finished_tasks()
            finished_jobs=finished_jobs+self.obj_infer[qqk].taskrun
            sum_infer = sum_infer+self.obj_infer[qqk].num_infer

            # task_cnt.append(len(self.obj_train[qqk].tasks))

        if round(clock.get_cur_time(),0) % 1 == 0:
            used_gpu_num_train = 0
            for ind_gpu in range(self.no_gpu_train):
                if len(self.obj_train[ind_gpu].tasks) > 0:
                    used_gpu_num_train = used_gpu_num_train+1

            used_gpu_num_infer = 0
            for ind_gpu in range(self.no_gpu_infer):
                if len(self.obj_infer[ind_gpu].tasks) > 0:
                    used_gpu_num_infer = used_gpu_num_infer+1

            # used_gpu_a = used_gpu_num_train*6
            # used_gpu_b = used_gpu_num_infer//5+used_gpu_num_infer
            # used_gpu_num=max(used_gpu_a,used_gpu_b)
            used_gpu_num = used_gpu_num_train+used_gpu_num_infer
            self.used_gpus.append(used_gpu_num)
            self.used_gpus_train.append(used_gpu_num_train)
            self.used_gpus_infer.append(used_gpu_num_infer)
        end_t = time.time()

        if round(clock.get_cur_time()-self.first_job_time,0) % self.log_rate == 0:
            self.num_day = self.num_day+1
            self.dump_data(self.num_day)
        if round(clock.get_cur_time(),0) % self.log_rate==0:
            print("***STime %.2fs, Loop %.2fs, GPUs(T/TS/I/IS/S/SS) %d/%d/%d/%d/%d/%d, Finished %d/%d, Train %d, Infer %d***"%(clock.get_cur_time()-\
                                                                                                                       self.task_list[0]['start_time'],\
                                                                                                                       end_t-start_t,\
                                                                                                                       used_gpu_num_train,\
                                                                                                                       self.no_gpu_train,\
                                                                                                                       used_gpu_num_infer,\
                                                                                                                       self.no_gpu_infer,\
                                                                                                                         self.no_gpu_train\
                                                                                                                         +self.no_gpu_infer,\
                                                                                                                               used_gpu_num,\
                                                                                                                       finished_jobs,\
                                                                                                                       self.no_jobs,\
                                                                                                                       sum_train,\
                                                                                                                       sum_infer))

    def dispatch_job_to_gpu(self, schedule_job_ind):
        success_job_ind = []
        for job_ind in schedule_job_ind:
            schedule = False
            if self.task_list[job_ind]['tagtask'] == 0: # train
                for gpu_ind in range(self.no_gpu_train):
                    #print(k)
                    x=self.obj_train[gpu_ind].assign_task(gpu_ind,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                    if(x==True):
                        self.initialschedulegpu.append(gpu_ind)
                        schedule=True
                        # infersum=infersum+self.obj_infer[gpu_ind].inferenceocount_sim
                        break
                    else:
                        schedule=False
                        # infersum=infersum+self.obj_train[gpu_ind].inferenceocount_sim
                if(schedule==False):
                    self.no_gpu_train = self.no_gpu_train+1
                    if self.no_gpu_train <= self.max_no_gpu_train:
                        self.obj_train.append(GPU(memory=self.gpumemory[self.no_gpu_train-1],\
                                             utilization=self.gpu_max_util,\
                                             dram=self.gpumemory[self.no_gpu_train-1],
                                             clock=self.clock,\
                                             train_time = self.train_time,\
                                                 ddl=self.ddl))
                        x=self.obj_train[-1].assign_task(self.no_gpu_train-1,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        assert x==True
                        schedule = True
                    else:
                        self.no_gpu_train = self.no_gpu_train-1
                        schedule = False
                        
            elif self.task_list[job_ind]['tagtask'] == 1: # infer:
                for gpu_ind in range(self.no_gpu_infer):
                    #print(k)
                    x=self.obj_infer[gpu_ind].assign_task(gpu_ind,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                    if(x==True):
                        self.initialschedulegpu.append(gpu_ind)
                        schedule=True
                        # infersum=infersum+self.obj_infer[gpu_ind].inferenceocount_sim
                        break
                    else:
                        schedule=False
                        # infersum=infersum+self.obj_infer[gpu_ind].inferenceocount_sim
                if(schedule==False):
                    self.no_gpu_infer = self.no_gpu_infer+1
                    if self.no_gpu_infer <= self.max_no_gpu_infer:
                        self.obj_infer.append(GPU(memory=self.gpumemory[self.no_gpu_infer-1],\
                                             utilization=self.gpu_max_util,\
                                             dram=self.gpumemory[self.no_gpu_infer-1],
                                             clock=self.clock,\
                                             train_time = self.train_time,\
                                                 ddl = self.ddl))
                        x=self.obj_infer[-1].assign_task(self.no_gpu_infer-1,self.task_list[job_ind],self.task_list[job_ind]['start_time'])
                        assert x==True
                        schedule = True
                    else:
                        self.no_gpu_infer = self.no_gpu_infer-1
                        schedule=False
            if schedule == True:
                success_job_ind.append(job_ind)
        return success_job_ind
    
    def deadlinemissscheduler(self,clock):
        self.plotinggpuassignment=[]
        missed_job=[]
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

        print("Waiting for tasks to complete")
        scheduled_flag=False
        while(scheduled_flag==False):
            clock.tic_job() # increase simulated time
            finished_jobs=0
            check_jobs = 0
            sum_train = 0
            sum_infer = 0
            start_t = time.time()
            for qqk in range(self.no_gpu_train):
                self.obj_train[qqk].remove_finished_tasks()
                finished_jobs=finished_jobs+self.obj_train[qqk].taskrun
                sum_train = sum_train+self.obj_train[qqk].num_train
                sum_infer = sum_infer+self.obj_train[qqk].num_infer
                check_jobs=check_jobs+len(self.obj_train[qqk].tasks)
                # task_cnt.append(len(self.obj_train[qqk].tasks))
            for qqk in range(self.no_gpu_infer):
                self.obj_infer[qqk].remove_finished_tasks()
                finished_jobs=finished_jobs+self.obj_infer[qqk].taskrun
                sum_infer = sum_infer+self.obj_infer[qqk].num_infer
                sum_train = sum_train+self.obj_infer[qqk].num_train
                check_jobs=check_jobs+len(self.obj_infer[qqk].tasks)
                # task_cnt.append(len(self.obj_train[qqk].tasks))

            if round(clock.get_cur_time(),0) % 1 == 0:
                used_gpu_num_train = 0
                for ind_gpu in range(self.no_gpu_train):
                    if len(self.obj_train[ind_gpu].tasks) > 0:
                        used_gpu_num_train = used_gpu_num_train+1
                used_gpu_num_infer = 0
                for ind_gpu in range(self.no_gpu_infer):
                    if len(self.obj_infer[ind_gpu].tasks) > 0:
                        used_gpu_num_infer = used_gpu_num_infer+1
                # used_gpu_a = used_gpu_num_train*6
                # used_gpu_b = used_gpu_num_infer//5+used_gpu_num_infer
                # used_gpu_num=max(used_gpu_a,used_gpu_b)
                used_gpu_num = used_gpu_num_train+used_gpu_num_infer
                self.used_gpus.append(used_gpu_num)
                self.used_gpus_train.append(used_gpu_num_train)
                self.used_gpus_infer.append(used_gpu_num_infer)
            end_t = time.time()
            
            if finished_jobs != (self.no_jobs-sum_train-sum_infer):
                print("finshed_jobs %d"%(finished_jobs))
                print("check jobs %d"%(self.no_jobs-sum_train-sum_infer))
                print("check jobs task len %d"%(self.no_jobs-check_jobs))
            assert finished_jobs == (self.no_jobs-sum_train-sum_infer)
            if round(clock.get_cur_time()-self.first_job_time,0) % self.log_rate == 0:
                self.num_day = self.num_day+1
                self.dump_data(self.num_day)
            if round(clock.get_cur_time(),0) % self.log_rate==0:
                print("***Time %.2fs, Loop %.2fs, GPUs(T/TS/I/IS/S/SS) %d/%d/%d/%d/%d/%d, Finished %d/%d, Train %d, Infer %d***"%(clock.get_cur_time()-\
                                                                                                                           self.task_list[0]['start_time'],\
                                                                                                                           end_t-start_t,\
                                                                                                                           used_gpu_num_train,\
                                                                                                                           self.no_gpu_train,\
                                                                                                                           used_gpu_num_infer,\
                                                                                                                           self.no_gpu_infer,\
                                                                                                                             self.no_gpu_train\
                                                                                                                             +self.no_gpu_infer,\
                                                                                                                                  used_gpu_num,\
                                                                                                                           finished_jobs,\
                                                                                                                           self.no_jobs,\
                                                                                                                           sum_train,\
                                                                                                                           sum_infer))
                
            if(finished_jobs==self.no_jobs):
                scheduled_flag=True
                self.num_day = self.num_day+1
                self.dump_data(self.num_day)
                break
            else:
                scheduled_flag=False

                # clock.tic_job() # increase simulated time
#         temp=[]
#         empt=[]
#         for i in range(self.no_gpus):
#             if self.objs[i].latency:
#                 temp.append(self.objs[i].latency)
#         count=0
#         for x in range(0,len(temp)):
#             for y in temp[x]:
#                 count=count+1
#                 empt.append({'Latency':y['Latency'],'Slno':y['Jobid'],'Utilization':y['Utilization']})

#         newlist = sorted(empt, key=lambda d: d['Slno'])
#         #print(newlist)
#         #print(newlist[1]['Utilization'])

#         count=0
#         temp1=0
#         delta=0.6

#         print("Finsihed iteration 1")
#         #scheduled.clear()

#         # for yyy in range (self.no_gpus):
#         #     self.objs[yyy].taskrun=0
#         #     self.objs[yyy].latency.clear()

#         final_time=(clock.get_cur_time()-start_time2)*10**3
#         print(f"final time ={final_time}")
#         self.cost=0
#         self.cost=self.cost+((self.no_gpus/2)*final_time) +(interimtime*self.no_gpus)
#         print(f"Final Cost ={self.cost}")
                     
    def dump_data(self,num_day):
        with open('data/used_gpus/used_gpus_slo_day%d'%(num_day), 'w') as fp:
            for item in self.used_gpus:
                fp.write("%s\n" % item)
        self.used_gpus.clear()
        
        with open('data/used_gpus_train/used_gpus_slo_day%d'%(num_day), 'w') as fp:
            for item in self.used_gpus_train:
                fp.write("%s\n" % item)
        self.used_gpus_train.clear()
        
        with open('data/used_gpus_infer/used_gpus_slo_day%d'%(num_day), 'w') as fp:
            for item in self.used_gpus_infer:
                fp.write("%s\n" % item)
        self.used_gpus_infer.clear()
        
        if len(self.ddl) > 0:
            miss_ddl_rate = 1-np.sum(self.ddl)/len(self.ddl)
        else:
            miss_ddl_rate = 0
        with open('data/miss_ddl/miss_ddl_slo%d_day%d'%(self.slo_factor*100,num_day), 'w') as fp:
            fp.write("%s\n" % miss_ddl_rate)
        self.ddl.clear()
        print("SLO Violation %f"%(miss_ddl_rate*100))
        
        with open('data/train_time/train_time_slo_day%d'%(num_day), 'w') as fp:
            for item in self.train_time:
                fp.write("%s\n" % item)
        self.train_time.clear()    
        
    '''
    The default function from the main function!!!!!!!!! comes here after scheduler
    '''
    def schedule(self):
        # utilization=100 # change later
        time=0
        gpuspecs =	{
        "V100":32,
        "K80": 24,
        "P100":16,
        }   
        for i in range(len(self.jobid)):
            self.task_list.append({'Jobid':self.jobid[i],\
                                   'memory':self.task_mem[i],\
                                   'utilization':self.task_gpu_util[i],\
                                   'duration':self.duration[i],\
                                   'slno':i,\
                                   'start_time':self.start_time[i], \
                                   'meet_ddl':True})
            # print("\njob tags:")
            # print(self.tagtask[i])
        self.task_list = sorted(self.task_list, key=lambda d: d['start_time']) # sort ascending order based on start time
        self.first_job_time = self.task_list[0]['start_time']
        for i in range(len(self.task_list)):
            self.task_list[i]['start_time'] = self.task_list[i]['start_time']-self.first_job_time
        self.first_job_time=0
        scheduler.gpu_list(self)
        scheduler.gpumemoryassignment(self)
        self.gpuutilassignment()
        print("Max Total no of GPUS =",self.max_no_gpus)
        print("Length of GPU MEMORY=",len(self.gpumemory))
        # print("Final Job =",self.finaljob)
        for i in range(len(self.jobid)):
            self.task_list[i]['tagtask'] = self.tagtask[i]
  


        count=0
        self.initiallatency=[]
        self.job_altered_latency=[]

        temp1=0
        # delta=0.6 # inintial latency increase rate
            
        # for i in range(self.no_jobs):
            #if(self.task_list[i]['tagtask']==1):
                    #  temp1=newlist[count]['Utilization']
                    # self.task_list[i]['duration']=self.task_list[i]['duration']+(temp1/100)*self.task_list[i]['duration']
                    # count=count+1
            #else:
            # self.task_list[i]['duration']=self.task_list[i]['duration']+(delta)*self.task_list[i]['duration']

        self.clock = tic_svc(cur_time=self.task_list[0]['start_time'],delta=1.0) # create universal clock
        self.obj_train = [GPU(memory=self.gpumemory[0],\
                         utilization=self.gpu_max_util,\
                         dram=self.gpumemory[0],\
                         train_time = self.train_time,\
                         clock=self.clock,\
                             ddl=self.ddl) for i in range(self.no_gpu_train)]
        
        self.obj_infer = [GPU(memory=self.gpumemory[0],\
                         utilization=self.gpu_max_util,\
                         dram=self.gpumemory[0],\
                         train_time = self.train_time,\
                         clock=self.clock,\
                             ddl=self.ddl) for i in range(self.no_gpu_infer)]
        #print(self.task_list[1])
        #print(objs[6])

        #scheduler.allocate(self,self.task_list,objs)
        print("No of jobs =",len(self.jobid))
        print("\n")
        #clrscr()
        latency=[]
        temp=[]
        tempid=[]
        if(self.deadline==0):
            print("regular scheduler is not enabled. I quit!")
            exit();

        else:
            print("Running scheduler with deadline misses")
            import time 
            s_time = time.time()
            ss_time = self.clock.get_cur_time()
            self.deadlinemissscheduler(self.clock)
            e_time = time.time()
            ee_time = self.clock.get_cur_time()
            
            print("Finished all jobs!Real Exe. Time %f s, Simu Exe. Time %.2f s"%(e_time-s_time, ee_time-ss_time))
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
            print("Finished all jobs!")
#             for i in range(self.no_gpus):
#                 if self.objs[i].latency:
#                     #print(self.objs[i].latency)
#                     temp.append(self.objs[i].latency)
                    
#             #print(temp)
#             #print(len(temp))
#             count=0
#             empt=[]

#             #for x in range(0,len(temp)):
#              #   for y in temp[x]:
#               #      print(y)
#                #     count=count+1
#                 #    #print(temp[x][y])

#             #print(count)
            
#             for x in range(0,len(temp)):
#                 for y in temp[x]:
#                     count=count+1
#                     empt.append({'Latency':y['Latency'],'Slno':y['Jobid']})

#             newlist = sorted(empt, key=lambda d: d['Slno'])
#             #print((newlist))

                    
#             print("Length of inference =",len(newlist))
#             for i in range(len(newlist)):
#                 latency.append(newlist[i]['Latency'])
#                 #tempid.append(newlist[i]['Slno']) 
#                 tempid.append(i) 
#                 #print(newlist[i]['Slno'])
#                 #print(newlist[i]['Latency'])

#             with open('data/used_gpus', 'w') as fp:
#                 for item in self.used_gpus:
#                     fp.write("%s\n" % item)
#             print('Used GPU written')
            
#             with open('data/latency', 'w') as fp:
#                 for item in latency:
#                     fp.write("%s\n" % item)
#             print('Latency written')


                
#             with open('data/tempid', 'w') as fp:
#                 for item in tempid:
#                     fp.write("%s\n" % tempid)
#             print('Job id written')



#            # with open('data/gpudatacenter', 'w') as fp:
#             #    for item in self.gpu_datacenter:
#              #       fp.write("%s\n" % item['Time'])
#             #print('timestamp written')

#             with open('data/gpudatatimestamp', 'w') as fp:
#                 for item in self.gpu_datacenter:
#                     fp.write("%s\n" % item)
#             print('gpudatacenter written')

#             with open('data/schedulingtimestamp', 'w') as fp:
#                 for item in self.timeschedulegpu:
#                     fp.write("%s\n" % item['Timestamp'])
#             print('timescheduling written')

#             with open('data/schedulingtimejob', 'w') as fp:
#                 for item in self.timeschedulegpu:
#                     fp.write("%s\n" % item['Jobid'])
#             print('Job written')



            




            