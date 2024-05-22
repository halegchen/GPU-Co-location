import os
import csv
import pandas as pd
from multiprocessing import Process
import threading
import numpy as np
import utils
import time
from scheduler import scheduler
import sys, getopt
import argparse
import random


submit_time=[]
duration=[]
jobid=[]
numinst=[]
num_cpu=[]
num_GPU=[]
gpu_util=[]
gpu_mem=[]
num_inst=[]
start_time=[]

#### reading the input dataset
def read(dataset):
    df = pd.read_csv(dataset,index_col=0,low_memory=False)
    print(df)
    duration[:]=df['runtime']
    #submit_time[:]=df['submit_time']
    jobid[:]=df['job_name']
    #numinst[:]=df['num_inst']
    #num_cpu[:]=df['num_cpu']  
    #num_GPU[:]=df['num_gpu']
    gpu_util[:]=df['gpu_wrk_util']
    gpu_mem[:]=df['max_gpu_wrk_mem']
    num_inst[:]=df['inst_num']
    start_time[:]=df['start_time_t']



####prints the input arguments for the function to be executed
def argumentprint(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--num_gpu", type=int, default=1, help="The number of GPUs")
    parser.add_argument("-c", "--config", type=int, default=2, help="Proposed Design(2)/Baselines(3-6)")
    parser.add_argument("-slo", "--slo_factor", type=float, default=1.0, help="SLO factor")
    parser.add_argument("-d", "--directory", type=str, default="../testrace.csv", help="Directory of trace data file (csv)")
    args = parser.parse_args()
    no_of_GPUs=args.num_gpu
    config=args.config
    dataset=args.directory
    slo_factor=args.slo_factor
     
    return no_of_GPUs, config,slo_factor,dataset


######main function
if __name__ == "__main__":
    no_of_GPUs=10
    deadlinemiss=10
    no_of_GPUs,configuration,slo_factor,dataset=argumentprint(sys.argv[1:])   
    print("slo factor %.2f"%slo_factor)
    print("GPU simulator")
    #print(dataset)
    read(dataset)
    
    # print("Job duration: ")
    # print(duration)
    # print(" ")
    
    print("No of jobs =",len(jobid))

    print("Starting Scheduling")
    print ('No of GPUs ', no_of_GPUs)
    print ('deadlinemisses ', configuration)
    # with open("infer_latency_vs_sleep.txt", "r") as fp:
    #     data=(fp.read()).splitlines()
    # print(len(data))
    len_task = len(gpu_util)
    for i in range(0,len(gpu_util)):
        gpu_cnt = gpu_util[i]//100
        gpu_cnt_mem = gpu_mem[i]//32
        if (gpu_util[i] > 100) and (gpu_cnt > 0) and (gpu_cnt >= gpu_cnt_mem):
            gpu_cnt = gpu_util[i]//100+1
            if gpu_cnt_mem > 0:
                first_gpu_mem = gpu_mem[i]-32*gpu_cnt_mem
                assert first_gpu_mem>0
            else:
                first_gpu_mem=(gpu_util[i]-100*(gpu_cnt-1))/gpu_util[i]*gpu_mem[i]
                assert (first_gpu_mem>0) and (first_gpu_mem<=32)
            for j in range(int(gpu_cnt-1)):
                new_duration =  duration[i]
                duration.append(new_duration)
                new_jobid = jobid[i]
                jobid.append(new_jobid)
                new_gpu_util = 100
                gpu_util.append(new_gpu_util)
                new_gpu_mem = (gpu_mem[i]-first_gpu_mem)/(gpu_cnt-1)
                gpu_mem.append(new_gpu_mem)
                new_num_inst = 1
                num_inst.append(new_num_inst)
                new_start_time = start_time[i]
                start_time.append(new_start_time)
            if (new_gpu_mem > 32) or (new_gpu_mem <= 0):
                print("Wrong Mem %f"%(new_gpu_mem))
                print("Util cnt %d, Util %f, Mem cnt %d, Mem %f"%(gpu_cnt, gpu_util[i], gpu_cnt_mem, gpu_mem[i]))
            gpu_mem[i] = first_gpu_mem
            temp_util = gpu_util[i]
            gpu_util[i] = gpu_util[i]-100*(gpu_cnt-1)
            if gpu_util[i]==0:
                print("Util cut wrong, gpu util %f, gpu_cnt %d gpu_cnt_mem %d"%(temp_util, gpu_cnt, gpu_cnt_mem))
        
        elif (gpu_mem[i] > 32) and (gpu_cnt_mem > 0) and (gpu_cnt < gpu_cnt_mem):
            gpu_cnt_mem = gpu_mem[i]//32+1
            first_gpu_util = (gpu_mem[i]-32*(gpu_cnt_mem-1))/gpu_mem[i]*gpu_util[i]
            assert (first_gpu_util > 0) and (first_gpu_util <= 100) 
            for j in range(int(gpu_cnt_mem-1)):
                new_duration =  duration[i]
                duration.append(new_duration)
                new_jobid = jobid[i]
                jobid.append(new_jobid)
                new_gpu_mem = 32
                gpu_mem.append(new_gpu_mem)
                new_gpu_util = (gpu_util[i]-first_gpu_util)/(gpu_cnt_mem-1)
                if new_gpu_util <= 0:
                    print("mem cut middle")
                gpu_util.append(new_gpu_util)
                new_num_inst = 1
                num_inst.append(new_num_inst)
                new_start_time = start_time[i]
                start_time.append(new_start_time)
            if new_gpu_util > 100:
                print("Wrong Util")
            gpu_util[i] = first_gpu_util
            if gpu_util[i] <= 0:
                print("mem cut final")
            gpu_mem[i] = gpu_mem[i]-32*(gpu_cnt_mem-1)      
            assert gpu_mem[i] > 0   
        
        # if(gpu_util[i]>100 and gpu_util[i]<170):
        #  gpu_util[i]=gpu_util[i]-75
        # elif(gpu_util[i]>170 and gpu_util[i]<1000):
        #  gpu_util[i]=99.99

        # if(gpu_mem[i]>32):
            # gpu_mem[i]=32
    print(f"Num of Tasks after processing {len(jobid)}")
    print("Max Util %f, Max Mem %f"%(np.max(gpu_util), np.max(gpu_mem)))
    jobs =len(jobid)
    # exit()
    # print("\nJob Name\t\tStart Time\tRuntime\tUtil\tMem")
    # for i in range(0,len(gpu_util)):
    #     print("%s\t%d\t%d\t%.3f\t%.2f"%(jobid[i],start_time[i],duration[i],gpu_util[i],gpu_mem[i]))
    #object of the simulator class
    x=scheduler(alloc_policy=0,\
                cluster=None,\
                no_gpus=int(no_of_GPUs),\
                no_jobs=jobs,\
                deadline=int(configuration),\
                submit_time=submit_time,\
                duration=duration,\
                jobid=jobid,\
                numinst=numinst, \
                num_cpu=num_cpu,\
                num_GPU=num_GPU,\
                task_mem=gpu_mem,\
                task_gpu_util=gpu_util,\
                num_inst=num_inst,\
                start_time=start_time,\
                control_period=25,\
               slo_factor=float(slo_factor))
    
    #execution of the simulator
    x.schedule()