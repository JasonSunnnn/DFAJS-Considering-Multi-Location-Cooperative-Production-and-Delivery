import instant_data
import pickle

import numpy as np


def loda_date(Scenario_index):
    # 载入数据
    problem_information = instant_data.load_instant(Scenario_index)

    return problem_information

if __name__ == '__main__':
    
    for Scenario_index1 in range(6):
        #Scenario_index=1
        #Scenario_index=50
        # 采用加权的方式找到最优解
        # Scenario_index=1
        weigh=[]
        runtimes=3 #>=3最好是3的倍数，因为目标数为3
        #总运行次数
        Scenario_index=Scenario_index1+3
        problem_information = loda_date(Scenario_index)
        
        n11=0
        for i in range(runtimes):
            for j in range(runtimes-i):
                n11+=1
        

        PF = []
        CPU_TIME = []
        GAP = []
        ModelSet=[]
        Code=[]
        n_operation = problem_information.sum_oper
        n_job=problem_information.n_job
        

        # result_name = 'result/case' + str(Scenario_index+1)+'工件数'+str(n_job) + '.pkl'
        # with open(result_name, 'rb') as f:
        #     result_loaded = pickle.load(f)
        # my=np.array(result_loaded['Solution'])
        # a=my.reshape(my.shape[0],3*n_operation)
        print('工序数=',n_operation)
        n1=0
        flag=0
       

        result_name = 'result/case' + str(Scenario_index+1)+'工件数'+str(n_job) + '.pkl'
        with open(result_name, 'rb') as f:
            result = pickle.load(f)
        a1=np.array(Code)
        a2=a1.reshape(a1.shape[0],3*n_operation)
        a3=np.array(GAP)
        pf=result['PF']
        a4=np.array(pf)
        np.savetxt('PF_case'+str(Scenario_index+1)+'.csv',a4, delimiter=',')
