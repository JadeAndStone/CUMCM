import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 设置全局宇体为中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 使用 SimHei 字体
plt.rcParams["axes.unicode_minus"] = False
# 正确显示负号

# 假设Excel文件名为'data.xlsx'
file_path = "1_university_employment_cleaned.xlsx"

# 创建图形
plt.figure(figsize=(12, 6))

# 定义从冷色到暖色的渐变颜色映射
# 2016(冷色:蓝绿色) -> 2022(暖色:橙红色)
cool_color = np.array([0, 0, 255]) / 255  # 蓝绿色
warm_color = np.array([255, 0, 0]) / 255  # 橙红色
# 生成7个过渡颜色
colors = [cool_color + (warm_color - cool_color) * i / 6 for i in range(7)]


# 遍历2016-2022年
for i, year in enumerate(range(2016, 2023)):
    try:
        # 读取对应sheet
        df = pd.read_excel(file_path, sheet_name=str(year))

        # 绘制该年份的数据
        plt.plot(
            df["日期"].dt.day_of_year,
            df["博士"],
            label=str(year),
            color=colors[i],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=5,
        )
    except Exception as e:
        print(f"处理{year}年数据时出错: {e}")

# 设置图形格式
plt.title("2016-2022年博士就业率趋势", fontsize=14)
plt.xlabel("日期", fontsize=12)
plt.ylabel("整体数值", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# 设置x轴为日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# 添加图例
plt.legend(title="年份", bbox_to_anchor=(1.05, 1), loc="upper left")

# 自动调整日期标签
plt.gcf().autofmt_xdate()

# 显示图形
plt.tight_layout()
plt.show()
