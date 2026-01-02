import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from ypstruct import structure

def load_instant(Scenario_index):
    information_index = 'instant/' + 'case' + str(Scenario_index+1) + '.csv'
    information = np.genfromtxt(information_index, dtype='int',delimiter=',')
    
    n_job=information[0,0]
    n_mach=information[0,1]
    sum_oper=information[0,2]
    n_factory=information[0,3]
    M_factory=information[1,:n_mach]
    n_operation_of_job=information[2,:n_job]
    #加工机器
    m=[]
    for i in range(n_job):
        m1=information[3+i]
        temp_id1=0
        m2=[]
        for ii in range(n_operation_of_job[i]):
            n_mach_temp=m1[temp_id1]
            temp_id1+=1
            m3=m1[temp_id1:temp_id1+n_mach_temp]
            temp_id1+=n_mach_temp
            m2.append(m3)
            #如果报错就说明问题信息不对，机器数和工序数对不上
        m.append(m2)
    temp_id=3+n_job

    #加工时间
    pt=[]
    for i in range(n_job):
        m1=information[temp_id+i]
        temp_id1=0
        m2=[]
        for ii in range(n_operation_of_job[i]):
            n_mach_temp=m1[temp_id1]
            temp_id1+=1
            m3=m1[temp_id1:temp_id1+n_mach_temp]
            temp_id1+=n_mach_temp
            m2.append(m3)
            #如果报错就说明问题信息不对，机器数和工序数对不上
        pt.append(m2)
    temp_id+=n_job

    #BOM
    bom=[]
    for i in range(n_job):
        m1=information[temp_id+i]
        temp_id1=0
        m2=[]
        for ii in range(n_operation_of_job[i]):
            n_mach_temp=m1[temp_id1]
            temp_id1+=1
            m3=m1[temp_id1:temp_id1+n_mach_temp]
            temp_id1+=n_mach_temp
            m2.append(m3)
            #如果报错就说明问题信息不对，机器数和工序数对不上
        bom.append(m2)
    temp_id+=n_job
    # TT & TC
    Tt=information[temp_id:temp_id+n_mach,:n_mach]
    temp_id+=n_mach
    Tc=information[temp_id:temp_id+n_mach,:n_mach]
    temp_id+=n_mach
    CTt=information[temp_id:temp_id+n_job,:n_factory]
    temp_id+=n_job
    CTc=information[temp_id:temp_id+n_job,:n_factory]
    temp_id+=n_job
    due_data_of_job=information[temp_id:temp_id+n_job,:1]


    problem_information = structure()
    problem_information.M = m
    problem_information.PT = pt
    problem_information.BOM = bom
    problem_information.TT = Tt
    problem_information.TC = Tc
    problem_information.product_TT = CTt
    problem_information.product_TC = CTc
    problem_information.due_data_of_job = due_data_of_job
    problem_information.n_machine = n_mach
    problem_information.n_job = n_job
    problem_information.sum_oper = sum_oper
    problem_information.n_factory=n_factory
    problem_information.n_operation_of_job = n_operation_of_job
    problem_information.M_factory=M_factory

    return problem_information


if __name__ == "__main__":
    load_instant