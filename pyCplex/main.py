from docplex.mp.model import Model

from docplex.util.environment import get_environment

import instant_data
import numpy as np
import assimble_job_shop
import json
import pickle
import pareto
from typing_extensions import Literal

def loda_date(Scenario_index):
    # 载入数据
    problem_information = instant_data.load_instant(Scenario_index)

    return problem_information

def build_model(problem_information,n_job, weigh=None):
    # n_job = problem_information.n_job
    # n_operation = problem_information.sum_oper
    #n_job=2
    n_operation = sum(problem_information.n_operation_of_job[:n_job])

    n_machine = problem_information.n_machine
    n_operation_of_job = problem_information.n_operation_of_job
    M = problem_information.M #机器集
    BOM = problem_information.BOM

    # 增加工件和机器的释放时间
    arrive_time_of_job = np.zeros(n_job,dtype='int')  
    arrive_time_of_machine = np.zeros(n_machine,dtype='int')

    PT = problem_information.PT
    TT = problem_information.TT
    TC = problem_information.TC
    M_factory=problem_information.M_factory
    product_TT=problem_information.product_TT
    product_TC=problem_information.product_TC
    n_factory=problem_information.n_factory
    due_data_of_job = problem_information.due_data_of_job
    due_data_of_job=[int(due_data_of_job[i]) for i in range(n_job)]

    #T = n_operation #需要优化T不然太慢了
    #优化T为各机器上最大的工序数
    M_temp=np.zeros(n_machine)
    for i in range(n_job):
        for j in range(n_operation_of_job[i]):
            for k in range(len(M[i][j])):
                M_temp[M[i][j][k]-1]+=1
    T=M_temp.max().astype('int')


    SM = [(k, t) for k in range(n_machine) for t in range(T)]
    FM = [(k, t) for k in range(n_machine) for t in range(T)]
    SO = [(i, j) for i in range(n_job) for j in range(n_operation_of_job[i])]
    FO = [(i, j) for i in range(n_job) for j in range(n_operation_of_job[i])]
    TCTEMP = [(i, j) for i in range(n_job) for j in range(n_operation_of_job[i])]  #记录工件工序的运输成本
    TTTEMP = [(i, j) for i in range(n_job) for j in range(n_operation_of_job[i])]  #记录工件工序的运输时间

    TT_val = [i for i in range(n_job)]
    TC_val = [i for i in range(n_job)]
    Y = 10000  # 一个十分大的数

    # 定义模型种类， 这里是混合整数规划“MIP"
    mdl = Model('MIP')  # mdl是“model" 的缩写

    # 定义变量

    # 构建决策变量 X[i][j][k][t] = 1 表示：第i个工件的第j道工序分配在可选加工机器集上的机器k的第t个事件点上
    X = []
    for i in range(n_job):
        for j in range(n_operation_of_job[i]):
            for k in range(len(M[i][j])):
                for t in range(T):
                    X.append((i, j, k, t))
    X = mdl.binary_var_dict(X, name='X')  # 决策变量：X[i] = 0表示该箱子没有用，X[i] = 1表示该箱子用了
    
    SM_v = mdl.continuous_var_dict(SM, lb=0, name='SM_v')  #开始时间
    FM_v = mdl.continuous_var_dict(FM, lb=0, name='FM_v')  #结束时间

    SO_v = mdl.integer_var_dict(SO, lb=0, name='SO_v')
    FO_v = mdl.integer_var_dict(FO, lb=0, name='FO_v')
   
    TC_Matrix=mdl.continuous_var_dict(TCTEMP, lb=0, name='TC_Matrix')
    TT_Matrix=mdl.continuous_var_dict(TTTEMP, lb=0, name='TT_Matrix')
    TT_val = mdl.continuous_var_dict(TT_val, lb=0, name='TT_val') #拖期
    TC_val = mdl.continuous_var_dict(TC_val, lb=0, name='TC_val') #运输成本

    ###########################################################定义目标函数
    makespan=0
    # 总完工最小
    obj1=mdl.continuous_var(makespan, name='Makespan' )

    # 总拖期最小
    obj2 = mdl.sum(TT_val[i] for i in range(n_job))

    # 总运费最小
    obj3 = mdl.sum(TC_val[i] for i in range(n_job))


    obj_number = 2
    if obj_number == 1:

        mdl.minimize(obj1)

    else:

        obj = weigh[0] * obj1 + weigh[1] * obj2+ weigh[2] * obj3
        #obj = obj1 +  obj2+  obj3
        mdl.minimize(obj)


    ###########################################################定义约束条件
    #################################################################################决策变量
    # 每个工序必须要分配到一个事件点上
    mdl.add_constraints(mdl.sum(X[i, j, k, t] for k in range(len(M[i][j])) for t in range(T)) == 1
                        for i in range(n_job) for j in range(n_operation_of_job[i]))

    # kk 表示第k台机器 k表示机器集中的第k台机器
    # 每个事件点至多分配一个工序
    mdl.add_constraints(mdl.sum(X[i, j, k, t] for i in range(n_job) for j in range(n_operation_of_job[i])
                                if kk + 1 in M[i][j] for k in range(len(M[i][j])) if M[i][j][k] == kk + 1) <= 1
                        for kk in range(n_machine) for t in range(T))

    # 前一个事件点分配了工序，后一个事件点才能分配工序
    mdl.add_constraints(mdl.sum(X[i, j, k, t] for i in range(n_job) for j in range(n_operation_of_job[i])
                                if kk + 1 in M[i][j] for k in range(len(M[i][j])) if M[i][j][k] == kk + 1) >=
                        mdl.sum(X[i, j, k, t + 1] for i in range(n_job) for j in range(n_operation_of_job[i])
                                if kk + 1 in M[i][j] for k in range(len(M[i][j])) if M[i][j][k] == kk + 1)
                        for kk in range(n_machine) for t in range(T - 1))
    #################################################################################目标约束
    # 计算完工时间
    # for i in range(n_job):
    #     j = n_operation_of_job[i] - 1
    #     mdl.add_constraint(obj1 >= FO_v[i, j] )
    for kk in range(n_machine):
        for t in range(T):
            mdl.add_constraint(obj1 >= FM_v[kk, t] )
    
    # 计算运输成本和时间
    # 先建立运输时间、成本矩阵，n_job行n_operation_of_job列
    # 工序的运输时间矩阵和成本矩阵等于BOM中子件运输到装配机器的时间和成本
    for i in range(n_job):
        for j2 in range(n_operation_of_job[i]):
            for l in range(len(BOM[i][j2])):
                j1=BOM[i][j2][l]-1 #j1是j2的子件
                for k1 in range(len(M[i][j1])):
                    for k2 in range(len(M[i][j2])):
                        k11=M[i][j1][k1]-1
                        k22=M[i][j2][k2]-1
                #此时 i j 为装配件，查其子件
                        mdl.add_constraint(TC_Matrix[i,j1] >= TC[k11,k22]*(mdl.sum(X[i, j1, k1, t1] for t1 in range(T))
                                               +mdl.sum(X[i, j2, k2, t2] for t2 in range(T))-1))
                        
                        mdl.add_constraint(TT_Matrix[i,j1] >= TT[k11,k22]*(mdl.sum(X[i, j1, k1, t1] for t1 in range(T))
                                                         +mdl.sum(X[i, j2, k2, t2] for t2 in range(T))-1))
                        # for t1 in range(T):
                        #     for t2 in range(T):
                        #         mdl.add_constraint(TC_Matrix[i,j1] >= TC[k11,k22]+Y*(X[i, j1, k1, t1] +X[i, j2, k2, t2]-2))
                        #         mdl.add_constraint(TT_Matrix[i,j1] >= TT[k11,k22]+Y*(X[i, j1, k1, t1] +X[i, j2, k2, t2]-2))
                        
                        # mdl.add_constraint(TT_Matrix[i,j1] >= TT[k11,k22]+Y*(mdl.sum(X[i, j1, k1, t1] +X[i, j2, k2, t2] 
                        #                                                            for t2 in range(T)
                        #                                                             for t1 in range(T))-2) )
                       
    #交付过程                                                                                                                 
    for i in range(n_job):
        j = n_operation_of_job[i]-1
        for k in range(len(M[i][j])):
            for t in range(T):
                f = M_factory[M[i][j][k]-1]-1
                mdl.add_constraint(TC_Matrix[i,j] >= product_TC[i,f]*X[i, j, k, t])
                mdl.add_constraint(TT_Matrix[i,j] >= product_TT[i,f]*X[i, j, k, t])


    for i in range(n_job):
        mdl.add_constraint(TC_val[i] >= mdl.sum(TC_Matrix[i,j]
                                                for j in range(n_operation_of_job[i])))

        
    # 计算工件拖期
    for i in range(n_job):
        j = n_operation_of_job[i] - 1
        for kk in range(n_machine):
            for t in range(T):
                if kk + 1 in M[i][j]:
                    for k in range(len(M[i][j])):
                        if M[i][j][k] == kk + 1:
                            mdl.add_constraint(TT_val[i] >= FM_v[kk, t] + Y * (X[i, j, k, t] - 1) +TT_Matrix[i,j] - due_data_of_job[i])

    # ## 用于检查解值
    # mdl.solve()
    # c=mdl.solution
    # d=obj2.solution_value
    # obj3.solution_value
    # # d=1
    # for i in range(n_job):
    #     j =n_operation_of_job[i]
    #     for l in range(len(BOM[i][j])):
    #         for k1 in range(len(M[i][BOM[i][j][l]])):
    #                 for k2 in range(len(M[i][j])):
                        
    #                     #此时 i j 为装配件，查其子件
    #                     a=sum(X[i, BOM[i][j][l], k1, t1].solution_value for t1 in range(T)) +sum(X[i, j, k2, t2].solution_value for t2 in range(T))-1
    #                     if a==1:
    #                         b=TC[M[i][BOM[i][j][l]][k1]-1,M[i][j][k2]-1]*a
    #                         if b!=0:
    #                             c=TC_Matrix[i,BOM[i][j][l]].solution_value
    #                     d=1



    #############################################################################顺序、时间变量##################################################
    # 工件的可以开始时间大于等于到达时间
    for kk in range(n_machine):
        for t in range(T):
            for i in range(n_job):
                for j in range(n_operation_of_job[i]):
                    if kk + 1 in M[i][j]:
                        for k in range(len(M[i][j])):
                            if M[i][j][k] == kk + 1:
                                mdl.add_constraint(SM_v[kk, t] >= arrive_time_of_job[i] + Y * (X[i, j, k, t] - 1))

    # BOM中上一道工序结束加工，后一道工序才能开始                                              
    for i in range(n_job):
        for j1 in range(n_operation_of_job[i]):
            for l in range(len(BOM[i][j1])):
                j2=BOM[i][j1][l]-1
                for kk1 in range(n_machine):
                    for k1 in range(len(M[i][j1])):
                        if M[i][j1][k1] == kk1 + 1:
                            for kk2 in range(n_machine):
                                if kk2 + 1 in M[i][j2]:
                                    for k2 in range(len(M[i][j2])):
                                        if M[i][j2][k2] == kk2 + 1:
                                            for t1 in range(T):
                                                for t2 in range(T):
                                                    mdl.add_constraint(FM_v[kk2, t2] + TT_Matrix[i,j2] <= SM_v[kk1, t1] + Y * (2-
                                                                X[i, j1, k1, t1] - X[i, j2, k2, t2]))
                # mdl.add_constraint(SO_v[i, j1] >= FO_v[i, j2] + TT_Matrix[i,j2] )
                #mdl.add_constraint(SO_v[i, j1] >= FO_v[i, j2] )   #测试
                                

    # 机器中上一个事件点加工结束，下一个事件点才能加工
    for kk in range(n_machine):
        for t in range(T - 1):
            mdl.add_constraint(FM_v[kk, t] <= SM_v[kk, t + 1])


    # 事件点的结束时间等于开始时间+加工时间
    for kk in range(n_machine):
        for t in range(T):
            #开始时间大于机器释放时间
            mdl.add_constraint(SM_v[kk, t] >= arrive_time_of_machine[kk])
            mdl.add_constraint(FM_v[kk, t] >= arrive_time_of_machine[kk])
            ## 两种约束代码
            # for i in range(n_job):
            #     for j in range(n_operation_of_job[i]):
            #         if kk + 1 in M[i][j]:
            #             for k in range(len(M[i][j])):
            #                 if M[i][j][k] == kk + 1:
            #                      mdl.add_constraint(FM_v[kk, t] >= SM_v[kk, t] + X[i, j, k, t] * PT[i][j][k])

            mdl.add_constraint(FM_v[kk, t] == SM_v[kk, t] + mdl.sum(X[i, j, k, t] * PT[i][j][k]
                                                                    for i in range(n_job) 
                                                                    for j in range(n_operation_of_job[i]) 
                                                                    if kk + 1 in M[i][j] 
                                                                    for k in range(len(M[i][j])) if M[i][j][k] == kk + 1))
            

 
    # 测试求解状态
    # 设置工作内存大小（以MB为单位）
    # mdl.parameters.workmem = 8*1024
    # mdl.parameters.mip.tolerances.mipgap=0.05
    # mdl.parameters.emphasis.mip=1
    # mdl.parameters.mip.strategy.heuristicfreq=5

    # mdl.solve()
    # c=mdl.solve_details
    # d1=obj1.solution_value
    # d2=obj2.solution_value
    # d3=obj3.solution_value

    # # 事件点的结束时间等于开始时间+加工时间
    # b=0
    # for kk in range(n_machine):
    #     t=T-1
    #     if FM_v[kk,t].solution_value>b:
    #         b=FM_v[kk,t].solution_value
    #         k1=kk
    # #建立机器时间矩阵
    # FM_f=np.zeros([n_machine,T])
    # SM_f=np.zeros([n_machine,T])
    # for kk in range(n_machine):
    #     for t in range(T):
    #         FM_f[kk,t]=FM_v[kk,t].solution_value
    #         SM_f[kk,t]=SM_v[kk,t].solution_value

    # #建立运输时间成本矩阵
    # TT_f=np.zeros([n_job,n_operation_of_job.max()])
    # TC_f=np.zeros([n_job,n_operation_of_job.max()])
    # for i in range(n_job):
    #     for j in range(n_operation_of_job[i]):
    #         TT_f[i,j]=TT_Matrix[i,j].solution_value
    #         TC_f[i,j]=TC_Matrix[i,j].solution_value
    # print('CPU time =',mdl.solve_details.time)
    # b=1



    return mdl,SM_v,FM_v,TT_val,TC_Matrix,TT_Matrix,X,obj1,obj2,obj3,T

    


if __name__ == '__main__':
    
    for Scenario_index1 in range(15):
        #Scenario_index=1
        #Scenario_index=50
        # 采用加权的方式找到最优解
        # Scenario_index=1
        Scenario_index=Scenario_index1
        weigh=[]
        runtimes=6 #>=3最好是3的倍数，因为目标数为3
        #总运行次数
        problem_information = loda_date(Scenario_index)
        
        import pickle

        pkl_path = 'result/case1工件数5.pkl'   # 按实际文件名改
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)

        PF = []
        CPU_TIME = []
        GAP = []
        ModelSet=[]
        Code=[]
        n_operation = problem_information.sum_oper
        n_job=problem_information.n_job
        n_machine=problem_information.n_machine

        # result_name = 'result/case' + str(Scenario_index+1)+'工件数'+str(n_job) + '.pkl'
        # with open(result_name, 'rb') as f:
        #     result_loaded = pickle.load(f)
        # my=np.array(result_loaded['Solution'])
        # a=my.reshape(my.shape[0],3*n_operation)
        print('工序数=',n_operation)
        n1=0
        flag=0
        W = [[1,     0,     0],
            [0,     1,     0],
            [0,     0,     1],
            [1/3, 1/3, 1/3],
            [2/3,   0, 1/3],
            [0,   2/3, 1/3]]
        

        for i in range(runtimes):
            i +=1
            if flag==1:
                break

            #进度
            n1+=1
            print('|案例', Scenario_index+1,'|工件数', n_job, '|进度', n1 / runtimes * 100, '%')


            weigh= W[i]
            print('权重=',weigh)
            #weigh=[0.3333333333,0,0.6666666666666666]

            mdl,SM_v,FM_v,TT_val,TC_Matrix,TT_Matrix,X,obj1,obj2,obj3,T = build_model(problem_information,n_job, weigh)

            time_limit = 600
            # 求解模型并显示
            if time_limit:
                mdl.set_time_limit(time_limit)
            mdl.parameters.workmem = 8*1024
            mdl.parameters.mip.tolerances.mipgap=0.05
            mdl.parameters.emphasis.mip=1
            mdl.parameters.mip.strategy.heuristicfreq=1
            mdl.solve()
            print('CPU time =',mdl.solve_details.time)
            print(mdl.solve_details)
            solution = mdl.solution

            # print('obj     = ' + str(solution.get_objective_value()))
            # print(solution.solve_details)
            if solution is None:
                flag=1
                break
            cpu_time = solution.solve_details.time
            
            gap = solution.solve_details.mip_relative_gap
            obj_value = solution.get_objective_value()
            #n_job = problem_information.n_job
            product_id_set = problem_information.product_id_set
            print
            ModelSet.append(mdl)
            PF.append([obj1.solution_value, obj2.solution_value,obj3.solution_value])
            CPU_TIME.append(cpu_time)
            GAP.append(gap)

            # 将X编译为整数编码 #三段编码
            chromosome1=np.zeros(n_operation,dtype='int')
            chromosome2=np.zeros(n_operation,dtype='int')
            chromosome3=np.zeros(n_operation,dtype='int')
            SO_v=np.zeros([n_job,n_operation],dtype='int')
            FO_v=np.zeros([n_job,n_operation],dtype='int')
            TT_v=np.zeros([n_job,n_operation],dtype='int')
            TC_v=np.zeros([n_job,n_operation],dtype='int')
            n=0
            n_operation_of_job=problem_information.n_operation_of_job
            M=problem_information.M
            for i1 in range(n_job):
                for j1 in range(n_operation_of_job[i1]):
                    for k1 in range (len(M[i1][j1])):
                        for t1 in range(T):
                            if X[i1,j1,k1,t1].solution_value==1:
                                chromosome1[n]=k1+1
                                kk=M[i1][j1][k1]-1
                                SO_v[i1,j1]=SM_v[kk,t1].solution_value
                                FO_v[i1,j1]=FM_v[kk,t1].solution_value
                                TT_v[i1,j1]=TT_Matrix[i1,j1].solution_value #运输时间矩阵
                                TC_v[i1,j1]=TC_Matrix[i1,j1].solution_value #运输成本矩阵
                                n+=1
            SO_v_save=SO_v.copy()
            FO_v_save=FO_v.copy()
            TT_val1=np.zeros([n_job],dtype='int')
            for i1 in range(n_job):
                TT_val1[i1]=TT_val[i1].solution_value
            seq = sorted(((i, j) for i in range(n_job)
                                    for j in range(n_operation_of_job[i])),
                            key=lambda x: SO_v[x[0], x[1]])
            # 转成染色体：只存 job 编号
            for order, (i, j) in enumerate(seq):
                global_idx = sum(n_operation_of_job[:i]) + j   # 全局工序序号

                chromosome2[order] = i + 1                   # job 编号（染色体2）
                chromosome3[global_idx] = order                # 全局工序索引（染色体3）
                
            chromosome=np.concatenate((chromosome1,chromosome2,chromosome3))
            chromosome=chromosome.reshape(1,3*n_operation)
            Code.append(chromosome)

        #取非支配解集        
        pf=pareto.pareto_set(PF)

        result = {
                'Solution':Code,
                'Obj':PF,
                'n_opeartion':n_operation,
                'CPU_time':CPU_TIME,
                'GAP':GAP,
                'SO_v':SO_v_save,
                'FO_v':FO_v_save
        }

        result_name = 'result/case' + str(Scenario_index+1)+'工件数'+str(n_job) + '.pkl'
        with open(result_name, 'wb') as f:
            pickle.dump(result, f)
        a1=np.array(Code)
        a2 = np.array(a1).squeeze(axis=1)
        a3=np.array(GAP)
        a4=np.array(PF)
        a5=np.array(CPU_TIME)
        np.savetxt('solution_case'+str(Scenario_index+1)+'.csv',a2, delimiter=',')
        np.savetxt('gap_case'+str(Scenario_index+1)+'.csv',a3, delimiter=',')
        np.savetxt('objValue_case'+str(Scenario_index+1)+'.csv',a4, delimiter=',')
        np.savetxt('time_case'+str(Scenario_index+1)+'.csv',a5, delimiter=',')

