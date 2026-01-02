import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from ypstruct import structure

def generate_schedul(problem_information,position):

    # 导入调度数据
    T = copy.deepcopy(problem_information.T)
    M = copy.deepcopy(problem_information.M)
    n_operation_of_job = copy.deepcopy(problem_information.n_operation_of_job)
    n_operation = copy.deepcopy(problem_information.n_operation)
    n_machine = copy.deepcopy(problem_information.n_machine)
    n_product = copy.deepcopy(problem_information.n_product)
    n_job_of_product = copy.deepcopy(problem_information.n_job_of_product)
    arrive_time = copy.deepcopy(problem_information.arrive_time)
    arrive_time_of_job = copy.deepcopy(problem_information.arrive_time_of_job)
    due_data = copy.deepcopy(problem_information.due_data)
    constraint_matrix = copy.deepcopy(problem_information.constraint_matrix)
    n_job = copy.deepcopy(problem_information.n_job)

    # 初始化调度过程参数
    # 初始化机器可以开始加工时间 累计加工时间 累计加工工序 加工能耗 空载能耗
    TM = []
    for i in range(4):
        TM.append([])
        for j in range(int(n_machine)):
            TM[i].append(0)
    TM = np.array(TM)

    # 初始化工件可以开始加工时间 总工序数 当前加工工序数
    TP = []
    for i in range(5):
        TP.append([])
        for j in range(int(n_job)):
            if i == 0:
                TP[i].append(arrive_time_of_job[j])  # 从这里限制工件的可以开始加工时间
            elif i == 1:
                TP[i].append(n_operation_of_job[j])
            elif i == 2:
                TP[i].append(1)
            elif i == 3:
                TP[i].append(0)
            elif i == 4:
                TP[i].append(0)
            pass
        pass
    TP = np.array(TP)


    # 初始化机器调度矩阵
    TSE = []
    TSE_num = []
    for i in range(int(n_machine)):
        for j in range(3):  # 工序号 开始加工时间 结束加工时间
            TSE.append([])
            pass
        TSE_num.append(0)
        pass

    TSE_num = np.array(TSE_num)
    # 初始化工件调度矩阵
    TSE_Job = []
    for i in range(int(n_operation)):
        for j in range(3):  # 工序号 开始加工时间 结束加工时间
            TSE_Job.append([])
            pass

    # 第一层解码：
    position1 = position[0:n_operation]
    MSM = []  # 机器选择矩阵
    PJT = []  # 加工时间矩阵
    index = 0
    for i in range(int(n_job)):
        MSM.append([])
        PJT.append([])
        for j in range(n_operation_of_job[i]):
            MSM[i].append(M[i][j][position1[index] - 1])
            PJT[i].append(T[i][j][position1[index] - 1])
            index = index + 1

    # 第二层解码：首先解码成101,201的形式
    position2 = position[n_operation : 2 * n_operation]
    position2_val = np.zeros(int(n_job))
    OSM = []  # 工序排序部分的决策顺序
    for i in range(n_operation):
        position2_val[position2[i] - 1] = position2_val[position2[i] - 1] + 1
        OSM.append(position2[i] * 100 + position2_val[position2[i] - 1])

    # 第二部分解码完成
    for i in range(n_operation):
        ID = OSM[i]
        ID_B = ID % 100  # 第几道工序
        ID_A = (ID - ID_B) / 100  # 第几号工件
        m = MSM[int(ID_A - 1)][int(ID_B - 1)]  # 选择的机器
        PT = PJT[int(ID_A - 1)][int(ID_B - 1)]  # 加工时间
        if ID_B == 1:
            pre_job_end_time_set = []
            for j in range(int(n_job)):
                if constraint_matrix[int(ID_A - 1)][j] == 1:
                    pre_job_end_time_set.append(TP[0][j])

            if len(pre_job_end_time_set) == 0:
                ETO = TP[0][int(ID_A - 1)]
            else:
                ETO = max(pre_job_end_time_set)

        else:
            ETO = TP[0][int(ID_A - 1)]

        # 机器上的待加工的工序进行排序
        FIC = 0  # 标识变量，若FIC == 0，说明没有插空，若FIC == 1 说明插空完成

        for kk in range(int(TSE_num[m - 1])):
            # 空隙的开始时间
            if kk == 0:  # 若是第一个空隙，空隙的开始时间为0
                IBT = 0
            else:  # 若不是第一个空隙，则空隙的开始时间为此机器上加工的第kk -1个工序的结束加工时间
                IBT = TSE[3 * (m - 1) + 2][kk - 1]
            # 空隙的结束时间
            IET = TSE[3 * (m - 1) + 1][kk]  # 空隙的结束时间为此机器上加工的第kk个工序的开始时间

            # 工件的实际开始加工时间
            BT = max(ETO, IBT)
            # 判断是否满足插空
            if BT + PT <= IET:
                ET = BT + PT
                # 更新TSE
                ID = ID_A * 100 + ID_B
                TSE[(m - 1) * 3 + 0].insert(kk, ID)
                TSE[(m - 1) * 3 + 1].insert(kk, BT)
                TSE[(m - 1) * 3 + 2].insert(kk, ET)

                # 更新TSE_num
                TSE_num[m - 1] = TSE_num[m - 1] + 1

                # 更新TM
                TM[1][m - 1] = TM[1][m - 1] + PT  # 总的加工时间
                TM[2][m - 1] = TM[2][m - 1] + 1  # 总的加工工序数
                TM[3][m - 1] = TM[3][m - 1] + 0  # 总的加工工序数

                # 更新TP
                TP[0][int(ID_A - 1)] = ET
                TP[2][int(ID_A - 1)] = TP[2][int(ID_A - 1)] + 1
                TP[3][int(ID_A - 1)] = TP[3][int(ID_A - 1)] + 0
                if ID_B == 1:
                    TP[4][int(ID_A - 1)] = BT

                # 更新TSE_Job
                ID_B = ID % 100  # 第几道工序
                ID_A = (ID - ID_B) / 100  # 第几号工件
                TSE_Job[int(ID_A - 1) * 3 + 0].append(ID)
                TSE_Job[int(ID_A - 1) * 3 + 1].append(BT)
                TSE_Job[int(ID_A - 1) * 3 + 2].append(ET)

                FIC = 1
                break
                pass

        if FIC == 0:  # 若没有插空完成
            ID = ID_A * 100 + ID_B
            ETM = TM[0][m - 1]  # 机器加工完上一道工序
            if ETO > ETM:
                BT = ETO
            else:
                BT = ETM
                pass
            ET = BT + PT
            '''
            # 绘制甘特图
            plt.plot([BT, ET, ET, BT,BT],[i, i, i + 1, i + 1,i],color='black')
            plt.fill([BT, ET, ET, BT], [i, i, i + 1, i + 1], color=my_color[int(ID_A)])
            plt.text(BT + PT/2, i + 0.5, str(int(ID)), fontsize=20, verticalalignment='center',horizontalalignment='center')

            # 暂停 1 秒
            plt.pause(0.1)
            '''
            # 更新TSE
            TSE[(m - 1) * 3 + 0].append(ID)
            TSE[(m - 1) * 3 + 1].append(BT)
            TSE[(m - 1) * 3 + 2].append(ET)

            # 更新TSE_num
            TSE_num[m - 1] = TSE_num[m - 1] + 1

            # 更新TM
            TM[0][m - 1] = ET  # 加工完此道工序的结束时间
            TM[1][m - 1] = TM[1][m - 1] + PT  # 总的加工时间
            TM[2][m - 1] = TM[2][m - 1] + 1  # 总的加工工序数

            # 更新TP
            TP[0][int(ID_A - 1)] = ET
            TP[2][int(ID_A - 1)] = TP[2][int(ID_A - 1)] + 1

            if ID_B == 1:
                TP[4][int(ID_A - 1)] = BT

            # 更新TSE_Job
            ID_B = ID % 100  # 第几道工序
            ID_A = (ID - ID_B) / 100  # 第几号工件
            TSE_Job[int(ID_A - 1) * 3 + 0].append(ID)
            TSE_Job[int(ID_A - 1) * 3 + 1].append(BT)
            TSE_Job[int(ID_A - 1) * 3 + 2].append(ET)
            pass

    # 产品拖期
    Tardness = 0
    index = 0
    for j in range(n_product):
        Tardness = Tardness + max(0, TP[0][int(index + n_job_of_product[j] - 1)] - due_data[j])
        index = index + n_job_of_product[j]

    if Tardness == 0:
        Tardness = 0.00000000001

    # 计算WIP  WIP由两部分组成：第一部分是零部件阶段：用产品的开始加工时间-各个零部件开始加工时间，第二部分最终产品的结束时间
    WIP1 = 0
    WIP2 = 0
    for i in range(n_job):
        is_pre = 0
        for j in range(n_job):
            if constraint_matrix[j][i] == 1:  # 零部件i是零件j的上级零件，则将零件j的开工时间是i的WIP的结束时间
                WIP1 = WIP1 + TSE_Job[j * 3 + 1][0] - TSE_Job[i * 3 + 1][0]
                is_pre = 1

        if is_pre == 0:
            WIP2 = WIP2 + TSE_Job[i * 3 + 2][-1] - TSE_Job[i * 3 + 1][0]
    WIP = WIP1 + WIP2
    solution_information = structure()
    solution_information.Tardness = Tardness
    solution_information.WIP = WIP
    solution_information.TSE = TSE
    solution_information.TSE_num = TSE_num
    solution_information.TSE_Job = TSE_Job
    solution_information.TP = TP

    while True:
        # 知识4：满足交期的条件下，尽可能晚的加工或者装配，从而使得产品的库存降低
        solution_information_move_right = move_right(solution_information,problem_information)
        WIP_move_right = solution_information_move_right.WIP
        if WIP_move_right == WIP:
            break
        WIP = WIP_move_right
        solution_information = solution_information_move_right



    return solution_information

def move_right(solution_information,problem_information):
    '''
    按照拖期，将甘特图上的工序，自右往左的从左往右移动
    '''
    TSE = solution_information.TSE
    TSE_num = solution_information.TSE_num
    TSE_Job = solution_information.TSE_Job

    T = copy.deepcopy(problem_information.T)
    M = copy.deepcopy(problem_information.M)
    n_operation_of_job = copy.deepcopy(problem_information.n_operation_of_job)
    n_operation = copy.deepcopy(problem_information.n_operation)
    n_machine = copy.deepcopy(problem_information.n_machine)
    n_product = copy.deepcopy(problem_information.n_product)
    n_job_of_product = copy.deepcopy(problem_information.n_job_of_product)
    arrive_time = copy.deepcopy(problem_information.arrive_time)
    due_data = copy.deepcopy(problem_information.due_data)
    constraint_matrix = copy.deepcopy(problem_information.constraint_matrix)
    n_job = copy.deepcopy(problem_information.n_job)
    job_product = copy.deepcopy(problem_information.job_product)
    # 从拖期最晚的产品的开始，首先查看是否拖期，如果没有拖期，就将其移动到刚刚好拖期的时间，然后，从其第一个零部件开始，以其产品的开始时间作为拖期，从左往右移动，最后，将所有的产品都往右移动。
    due_data_sort = np.argsort(due_data)

    #gantt_chart_of_machine(TSE, TSE_num, problem_information)

    for i in range(n_product):
        product_id = int(due_data_sort[-i - 1])
        due_data_of_product = due_data[product_id]
        # 产品开始右移，右移的时候，有两个约束，第一个约束为产品自身的约束，最后一道工序的结束时间必须往后移动的不能大于拖期
        # 第二个约束为机器的约束，结束时间不能大于后一道工序的开始加工时间
        # 产品最后一道工序的最晚结束为拖期
        # 最后一道工序开始往后移
        JBT = due_data_of_product
        ID_A = job_product[product_id][1] + 1   # 工件号
        ID_B = n_operation_of_job[int(ID_A - 1)]  # 工序号
        ID = 100 * ID_A + ID_B
        for jj in range(n_operation_of_job[int(ID_A - 1)]):
            # 查找工序ID的位置
            find_ID = 0
            for j in range(n_machine):
                for k in range(TSE_num[j]):
                    if ID == TSE[j * 3][k]:
                        find_ID = 1
                        # 机器约束：该工序不是机器上的最后一道工序， 该产品的结束时间必须大于机器后一道工序的开始时间
                        # 工序约束：工序的结束时间必须小于工件上一道工序的结束时间
                        if k == TSE_num[j] - 1:
                            ET = TSE[j * 3 + 2][k]
                            if ET < JBT:  # 产品的拖期大于产品的完工时间，开始往右移动
                                PT = TSE[j * 3 + 2][k] - TSE[j * 3 + 1][k]
                                ET = JBT
                                BT = ET - PT

                                TSE[j * 3 + 2][k] = ET
                                TSE[j * 3 + 1][k] = BT

                                ID_B = ID % 100  # 第几道工序
                                ID_A = (ID - ID_B) / 100  # 第几号工件
                                TSE_Job[int(ID_A - 1) * 3 + 1][int(ID_B - 1)] = BT
                                TSE_Job[int(ID_A - 1) * 3 + 2][int(ID_B - 1)] = ET

                                JBT = TSE[j * 3 + 1][k]
                                ID = ID - 1
                            else:
                                JBT = TSE[j * 3 + 1][k]
                                ID = ID - 1
                        else:
                            ET = TSE[j * 3 + 2][k]
                            MBT = TSE[j * 3 + 1][k + 1]  # 机器上最后一道工序的结束时间
                            if ET < min(JBT,MBT):
                                PT = TSE[j * 3 + 2][k] - TSE[j * 3 + 1][k]
                                ET = min(JBT,MBT)
                                BT = ET - PT
                                TSE[j * 3 + 2][k] = ET
                                TSE[j * 3 + 1][k] = BT

                                ID_B = ID % 100  # 第几道工序
                                ID_A = (ID - ID_B) / 100  # 第几号工件
                                TSE_Job[int(ID_A - 1) * 3 + 1][int(ID_B - 1)] = BT
                                TSE_Job[int(ID_A - 1) * 3 + 2][int(ID_B - 1)] = ET

                                JBT = TSE[j * 3 + 1][k]
                                ID = ID - 1
                            else:
                                JBT = TSE[j * 3 + 1][k]
                                ID = ID - 1
                        break
                if find_ID == 1:
                    break
        PJBT = JBT
        # 零部件开始往后移
        for ii in range(len(job_product[product_id][0])):
            JBT = PJBT
            # 最后一道工序开始往后移
            ID_A = job_product[product_id][0][ii] + 1  # 工件号
            ID_B = n_operation_of_job[int(ID_A - 1)]  # 工序号
            ID = 100 * ID_A + ID_B
            for jj in range(n_operation_of_job[int(ID_A - 1)]):
                # 查找工序ID的位置
                find_ID = 0
                for j in range(n_machine):
                    for k in range(TSE_num[j]):
                        if ID == TSE[j * 3][k]:
                            find_ID = 1
                            # 机器约束：该工序不是机器上的最后一道工序， 该产品的结束时间必须大于机器后一道工序的开始时间
                            # 工序约束：工序的结束时间必须小于工件上一道工序的结束时间
                            if k == TSE_num[j] - 1:
                                ET = TSE[j * 3 + 2][k]
                                if ET < JBT:  # 产品的拖期大于产品的完工时间，开始往右移动
                                    PT = TSE[j * 3 + 2][k] - TSE[j * 3 + 1][k]
                                    ET = JBT
                                    BT = ET - PT

                                    TSE[j * 3 + 2][k] = ET
                                    TSE[j * 3 + 1][k] = BT

                                    ID_B = ID % 100  # 第几道工序
                                    ID_A = (ID - ID_B) / 100  # 第几号工件
                                    TSE_Job[int(ID_A - 1) * 3 + 1][int(ID_B - 1)] = BT
                                    TSE_Job[int(ID_A - 1) * 3 + 2][int(ID_B - 1)] = ET

                                    JBT = TSE[j * 3 + 1][k]
                                    ID = ID - 1
                                else:
                                    JBT = TSE[j * 3 + 1][k]
                                    ID = ID - 1
                            else:
                                ET = TSE[j * 3 + 2][k]
                                MBT = TSE[j * 3 + 1][k + 1]  # 机器上最后一道工序的结束时间
                                if ET < min(JBT, MBT):
                                    PT = TSE[j * 3 + 2][k] - TSE[j * 3 + 1][k]
                                    ET = min(JBT, MBT)
                                    BT = ET - PT
                                    TSE[j * 3 + 2][k] = ET
                                    TSE[j * 3 + 1][k] = BT

                                    ID_B = ID % 100  # 第几道工序
                                    ID_A = (ID - ID_B) / 100  # 第几号工件
                                    TSE_Job[int(ID_A - 1) * 3 + 1][int(ID_B - 1)] = BT
                                    TSE_Job[int(ID_A - 1) * 3 + 2][int(ID_B - 1)] = ET

                                    JBT = TSE[j * 3 + 1][k]
                                    ID = ID - 1
                                else:
                                    JBT = TSE[j * 3 + 1][k]
                                    ID = ID - 1

                            break
                    if find_ID == 1:
                        break

    # 更新TP矩阵
    TP = solution_information.TP
    for i in range(n_job):
        TP[0][i] = TSE_Job[i * 3 + 2][-1]
        TP[4][i] = TSE_Job[i * 3 + 1][0]

    # 产品拖期
    Tardness = 0
    index = 0
    for j in range(n_product):
        Tardness = Tardness + max(0, TP[0][int(index + n_job_of_product[j] - 1)] - due_data[j])
        index = index + n_job_of_product[j]

    # 计算WIP  WIP由两部分组成：第一部分是零部件阶段：用产品的开始加工时间-各个零部件开始加工时间，第二部分最终产品的结束时间
    WIP1 = 0
    for i in range(n_job):
        is_pre = 0
        for j in range(n_job):
            if constraint_matrix[j][i] == 1:  # 零部件i是零件j的上级零件，则将零件j的开工时间是i的WIP的结束时间
                WIP1 = WIP1 + TSE_Job[j * 3 + 1][0] - TSE_Job[i * 3 + 1][0]
                is_pre = 1
        if is_pre == 0:
            WIP1 = WIP1 + TSE_Job[i * 3 + 2][-1] - TSE_Job[i * 3 + 1][0]

    solution_information.Tardness = Tardness
    solution_information.WIP = WIP1
    solution_information.TSE = TSE
    solution_information.TSE_num = TSE_num
    solution_information.TSE_Job = TSE_Job

    return solution_information

def gantt_chart_of_machine(solution_information,problem_information,name):
    '''

    Args:
        solution_information:
        problem_information:
        name:

    Returns:

    '''
    mycolor = pd.read_excel('颜色.xlsx')
    my_color = np.array(mycolor)

    TSE = solution_information.TSE
    TSE_num = solution_information.TSE_num

    Tardness = solution_information.Tardness
    WIP = solution_information.WIP

    n_machine = copy.deepcopy(problem_information.n_machine)
    job_product = problem_information.job_product
    makspan = 0
    for i in range(n_machine):
        for j in range(TSE_num[i]):
            ID = TSE[3 * i + 0][j]
            ID_B = ID % 100  # 第几道工序
            ID_A = (ID - ID_B) / 100  # 第几号工件
            BT = TSE[3 * i + 1][j]
            ET = TSE[3 * i + 2][j]
            PT = ET - BT
            # 找到颜色
            for k in range(len(job_product)):
                if ID_A - 1 in job_product[k][0] or ID_A - 1 == job_product[k][1]:
                    color_ID = k
                    break
            plt.plot([BT, ET, ET, BT, BT], [i, i, i + 1, i + 1, i], color='black')
            plt.fill([BT, ET, ET, BT], [i, i, i + 1, i + 1], color=my_color[k,:])
            plt.text(BT + PT / 2, i + 0.5, str(int(ID)), fontsize=10, verticalalignment='center',
                     horizontalalignment='center')

            makspan = max(makspan, ET)

    TEXT = 'Tardness' + '  ' + str(int(Tardness))
    plt.text(makspan/8, n_machine, TEXT, fontsize=10, verticalalignment='center',
             horizontalalignment='center')

    TEXT = 'WIP' + '  ' + str(int(WIP))
    plt.text(makspan/2, n_machine, TEXT, fontsize=10, verticalalignment='center',
             horizontalalignment='center')


    plt.show()

def gantt_chart_of_job(TSE_Job,problem_information):

    mycolor = pd.read_excel('颜色.xlsx')
    my_color = np.array(mycolor)

    n_job = copy.deepcopy(problem_information.n_job)
    n_operation_of_job = copy.deepcopy(problem_information.n_operation_of_job)

    for i in range(n_job):
        for j in range(n_operation_of_job[i]):
            ID = TSE_Job[3 * i + 0][j]
            ID_B = ID % 100  # 第几道工序
            ID_A = (ID - ID_B) / 100  # 第几号工件
            BT = TSE_Job[3 * i + 1][j]
            ET = TSE_Job[3 * i + 2][j]
            PT = ET - BT
            plt.plot([BT, ET, ET, BT, BT], [i, i, i + 1, i + 1, i], color='black')
            plt.fill([BT, ET, ET, BT], [i, i, i + 1, i + 1], color=my_color[int(ID_A)])
            plt.text(BT + PT / 2, i + 0.5, str(int(ID)), fontsize=20, verticalalignment='center',
                     horizontalalignment='center')
    plt.show()


def repair_chromosome(problem_information,chrom):
    # 判断第k个工件是否是第j个工件的前一道工件，若是前一道工序则problem_information.constraint_matrix[j][k] == 1,在工序部分的编码中不能出现k在j后面
    # 若出现k在j后面，则需要将k与第一次出现j的位置进行交换，从而满足k在j前面
    chrom_B = chrom[problem_information.n_operation:2 * problem_information.n_operation]

    while True:
        repair = False
        for j in range(problem_information.n_operation):
            for k in range(j, problem_information.n_operation):
                if problem_information.constraint_matrix[chrom_B[j] - 1][chrom_B[k] - 1] == 1:
                    chrom_B_val = chrom_B[k]
                    chrom_B[k] = chrom_B[j]
                    chrom_B[j] = chrom_B_val
                    repair = True
                    break
            if repair == True:
                break
        if j == problem_information.n_operation - 1:
            break
    chrom[problem_information.n_operation:2 * problem_information.n_operation] = chrom_B

    return chrom