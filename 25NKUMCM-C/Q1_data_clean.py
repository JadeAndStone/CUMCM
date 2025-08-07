import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

original_data = pd.read_excel('1_university_employment.xlsx', sheet_name=None, header=None)

for sheet_name, df in original_data.items():

    df.columns = df.iloc[0]  # 设置第一行作为列名
    df = df[1:]  # 删除第一行
    df.reset_index(drop=True, inplace=True)  # 重置索引
    
    df['年份'] = sheet_name  # 添加年份列


    print(f"Sheet: {sheet_name}")
    print(df.head())

