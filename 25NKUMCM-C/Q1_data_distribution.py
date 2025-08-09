import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

#设置全局宇体为中文
plt.rcParams ['font.sans-serif'] = ['SimHei']
#使用 SimHei 字体
plt.rcParams ['axes.unicode_minus'] = False
#正确显示负号

# 假设Excel文件名为'data.xlsx'
file_path = '1_university_employment_cleaned.xlsx'

# 创建图形
plt.figure(figsize=(12, 6))

# 定义从冷色到暖色的渐变颜色映射
# 2016(冷色:蓝绿色) -> 2022(暖色:橙红色)
cool_color = np.array([0, 0, 255])/255  # 蓝绿色
warm_color = np.array([255, 0, 0])/255  # 橙红色
# 生成7个过渡颜色
colors = [cool_color + (warm_color - cool_color) * i/6 for i in range(7)]


# 遍历2016-2022年
for i, year in enumerate(range(2016, 2023)):
    try:
        # 读取对应sheet
        df = pd.read_excel(file_path, sheet_name=str(year))
        
        
        # 绘制该年份的数据
        plt.plot(df['日期'].dt.day_of_year, df['整体'], 
                 label=str(year), 
                 color=colors[i],
                 marker='o',
                 linestyle='-',
                 linewidth=2,
                 markersize=5)
    except Exception as e:
        print(f"处理{year}年数据时出错: {e}")

# 设置图形格式
plt.title('2016-2022年整体数据趋势', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('整体数值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置x轴为日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# 添加图例
plt.legend(title='年份', bbox_to_anchor=(1.05, 1), loc='upper left')

# 自动调整日期标签
plt.gcf().autofmt_xdate()

# 显示图形
plt.tight_layout()
plt.show()



# 创建春节日期字典（2016-2026年）
spring_festival_dates = {
    2016: datetime(2016, 2, 8),
    2017: datetime(2017, 1, 28),
    2018: datetime(2018, 2, 16),
    2019: datetime(2019, 2, 5),
    2020: datetime(2020, 1, 25),
    2021: datetime(2021, 2, 12),
    2022: datetime(2022, 2, 1),
    2023: datetime(2023, 1, 22),
    2024: datetime(2024, 2, 10),
    2025: datetime(2025, 1, 29),
    2026: datetime(2026, 2, 17)
}

def get_lunar_day_of_year(date):
    """计算日期在其农历年中的第几天"""
    year = date.year
    
    # 检查是否需要使用前一年的春节日期
    if date < spring_festival_dates[year]:
        #spring_festival = spring_festival_dates[year - 1]
        return np.nan
    else:
        spring_festival = spring_festival_dates[year]
    
    return (date - spring_festival).days

# 遍历2016-2022年
for i, year in enumerate(range(2016, 2023)):
    try:
        # 读取对应sheet
        df = pd.read_excel(file_path, sheet_name=str(year))
        
        # 绘制该年份的数据
        plt.plot(df['日期'].apply(get_lunar_day_of_year), df['整体'], 
                 label=str(year), 
                 color=colors[i],
                 marker='o',
                 linestyle='-',
                 linewidth=2,
                 markersize=5)
    except Exception as e:
        print(f"处理{year}年数据时出错: {e}")

# 设置图形格式
plt.title('2016-2022年整体数据趋势', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('整体数值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置x轴为日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# 添加图例
plt.legend(title='年份', bbox_to_anchor=(1.05, 1), loc='upper left')

# 自动调整日期标签
plt.gcf().autofmt_xdate()

# 显示图形
plt.tight_layout()
plt.show()