
def is_dominates(a, b):
    """
    判断解a是否支配解b。
    
    参数:
    a, b -- 两个解，格式为(list 或 tuple)，包含多个目标的值。
    
    返回:
    True -- 如果a支配b。
     False-- 如果a不支配b。
    """
    # 检查a是否在所有目标上都不比b差，并且在至少一个目标上比b好
    dominates = all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
    return dominates

def pareto_set(pareto_set):
    """
    从解集中筛选出帕累托解集。
    
    参数:
    solutions -- 一个包含多个解的列表，每个解是一个包含目标值的列表或元组。
    
    返回:
    pareto_set -- 帕累托解集的列表。
    """
    pf=pareto_set.copy()
    # 对于列表中的每个解，检查是否有其他解支配它
    for i, sol_i in enumerate(pareto_set):
        for sol_j in pareto_set:
            if is_dominates(sol_j, sol_i) and sol_j != sol_i:
                # 如果找到支配它的解，则从帕累托解集中移除它
                pf.remove(sol_i)
                break
    
    return pf