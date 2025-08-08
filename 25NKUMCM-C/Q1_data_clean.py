import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

original_data = pd.read_excel('1_university_employment.xlsx', sheet_name=None, header=None)

for sheet_name, df in original_data.items():

    data=df.copy()  # 复制数据以避免修改原始数据

    data.columns = data.iloc[0]  # 设置第一行作为列名
    data = data[1:]  # 删除第一行
    data.reset_index(drop=True, inplace=True)  # 重置索引
    data['年份'] = sheet_name  # 添加年份
    column_map = {
        '时间': '日期',
        '本科就业率': '本科',
        '硕士就业率': '硕士',
        '博士就业率': '博士',
        '高职就业率': '高职',
        '整体就业率': '整体',
        '备注': '备注',
        '本科生': '本科',
        '硕士研究生': '硕士',
        '博士研究生': '博士',
        '所有毕业生': '整体',
        '全体毕业生': '整体'
    }
    data.rename(columns=column_map, inplace=True)    

    if int(sheet_name) not in [2016, 2017, 2018]:
        data['高职'] = np.nan

    if int(sheet_name) != 2018:
        def dataconvert(point):
            
        data['日期']
        data['日期'] = pd.to_datetime(['日期'], errors='coerce')


    print(f"Sheet: {sheet_name}")
    print(data.head())

