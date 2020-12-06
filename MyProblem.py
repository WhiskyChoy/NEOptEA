  
# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import os
import pandas as pd
import math
from datetime import datetime

"""
    该案例展示了一个带等式约束的连续型决策变量最大化目标的单目标优化问题。
    该函数存在多个欺骗性很强的局部最优点。
    max f = 4*x1 + 2*x2 + x3
    s.t.
    2*x1 + x2 - 1 <= 0
    x1 + 2*x3 - 2 <= 0
    x1 + x2 + x3 - 1 == 0
    0 <= x1,x2 <= 1
    0 < x3 < 2
"""

n = 10
T = 1000
ep = 0.9
range_index = range(0, n)

if not os.path.exists('data'):
    os.mkdir('data')

if not os.path.exists('result'):
    os.mkdir('result')

c_max_csv = open('data/c_max.csv')
d_csv = open('data/d.csv')
s_csv = open('data/s.csv')

df_c_max = pd.read_csv(c_max_csv, header=None, index_col=None).values
df_d_csv = pd.read_csv(d_csv, header=None, index_col=None).values
df_s_csv = pd.read_csv(s_csv, header=None, index_col=0)

def get_s_arr(date: datetime):
    str_date = f'{date.year}/{date.month}/{date.day}'
    return df_s_csv.loc[str_date].values

s_arr = get_s_arr(datetime(2020, 4, 16))
s = {i: s_arr[i] for i in range_index}

def index_trans(i, j):
    if i > j:
        return i*n+j-i
    if i < j:
        return i*n+j-i-1


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 90  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim # 决策变量下界
        ub = [df_c_max[i][j]
             for i in range_index for j in range_index if i != j]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        d = [df_d_csv[i][j]
         for i in range_index for j in range_index if i != j]
        pop.ObjV = np.sum(Vars * d, 1, keepdims=True)        
        # 采用可行性法则处理约束
        pop.CV = np.hstack([ (s[i] * Vars[:,[index_trans(i,j)]] - s[j] *
                Vars[:,[index_trans(j,i)]] + T * math.log((1-ep)/ep)) for i in range_index
            for j in range_index
            if i != j])

