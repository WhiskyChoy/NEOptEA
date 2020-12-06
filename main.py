# -*- coding: utf-8 -*-
import geatpy as ea  # import geatpy
from MyProblem import MyProblem  # 导入自定义问题接口
import pandas as pd
import numpy as np
import os
from common import data_prefix, result_prefix, output_prefix
from datetime import datetime, time, timedelta


def run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc):
    if not os.path.exists(data_prefix):
        os.mkdir(data_prefix)

    if not os.path.exists(result_prefix):
        os.mkdir(result_prefix)

    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    date_str = date.strftime('%Y-%m-%d')

    output_dir = f'date={date_str} n={n} T={T} ep={ep} desc={desc}.csv'
    print(output_dir)

    range_index = range(0, n)
    c_max_csv = open(f'{data_prefix}/{c_max_dir}')
    d_csv = open(f'{data_prefix}/{d_dir}')
    s_csv = open(f'{data_prefix}/{s_dir}')

    df_c_max = pd.read_csv(c_max_csv, header=None, index_col=None).values
    df_d = pd.read_csv(d_csv, header=None, index_col=None).values

    df_s_csv = pd.read_csv(s_csv, header=None, index_col=0)
    str_date = f'{date.year}/{date.month}/{date.day}'
    s_arr = df_s_csv.loc[str_date].values
    df_s = {i: s_arr[i] for i in range_index}

    """================================实例化问题对象==========================="""
    problem = MyProblem(n, T, ep, df_c_max, df_d, df_s)  # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'RI'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes,
                      problem.ranges, problem.borders)  # 创建区域描述器
    # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population = ea.Population(Encoding, Field, NIND)
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(
        problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 500  # 最大进化代数
    myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = 0.7  # 重组概率
    myAlgorithm.logTras = 0  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    # BestIndi.save()  # 把最优个体的信息保存到文件中
    """==================================输出结果=============================="""
    print('Number of evaluation：%s' % myAlgorithm.evalsNum)
    print('Time used in seconds %s' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('Best objective value：%s' % BestIndi.ObjV[0][0])
        final_table = np.zeros((10, 10))
        count = 0
        for i in range(10):
            for j in range(10):
                if i != j:
                    final_table[i][j] = BestIndi.Phen[0, count]
                    count += 1
        df = pd.DataFrame(final_table, columns=None, index=None)
        df.to_csv(f'{output_prefix}/{output_dir}',
                  header=False, index=False, mode='w')
    else:
        print('No feasible solution!')


if __name__ == '__main__':
    n = 10
    T = 1000
    ep = 0.9

    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc = 'default'

    start_date = datetime(2020, 2, 16)
    end_date = datetime(2020, 4, 16)
    step = timedelta(days=1)
    total_steps = (end_date - start_date) // step + 1

    for i in range(0, total_steps):
        date = start_date + i * step
        run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc)
