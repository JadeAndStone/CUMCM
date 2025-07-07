import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
raw_data=pd.read_csv('2024E/file2.csv',encoding='gbk')
time=raw_data['时间'].copy()
# print(time.shape[0])
for i in range(time.shape[0]):
    time.iloc[i]=time.iloc[i][:10]
# time.head(10)
time.name='day'

raw_data=pd.concat([raw_data,time],axis=1)
index=raw_data.groupby('交叉口').groups['经中路-纬中路']
data=raw_data.iloc[index]
begin_day=datetime.date(2024,4,1)
end_day=datetime.date(2024,5,6)
day_sep=(end_day-begin_day).days+1
avg_num_list=np.zeros((day_sep,96))
j=0
while begin_day<=end_day:
    day_data=raw_data.iloc[data.groupby('day').groups[str(begin_day)]]
    day_data=day_data.sort_values('时间')
    # print(day_data)
    # print(float(day_data.iloc[26008]['时间'][11:][:2])/60)
    # print(day_data.shape)
    time_list=[]
    for i in range(day_data.shape[0]):
        time_str=day_data.iloc[i]['时间'][11:]
        time_num=float(time_str[:2])*60+float(time_str[3:5])+float(time_str[6:])/60
        time_list.append(time_num)
    # print(len(time_list))
    # print(pd.Series(time_list,name='time'))
    day_data=pd.concat([day_data.reset_index(),pd.Series(time_list,name='time')],axis=1)
    # print(day_data.shape[0])
    num_list=[]
    flag=1
    t=0
    for i in range(day_data.shape[0]):
        if day_data.iloc[i]['time']>flag*15:
            flag+=1
            num_list.append(t)
            t=0
        if i==day_data.shape[0]-1:
            num_list.append(t+1)
        t+=1
    # print(avg_num_list)
    avg_num_list[j]=np.array(num_list)
    j+=1
    begin_day+=datetime.timedelta(days=1)
# print(avg_num_list)
# avg_num_list=[avg_num_list[i]/(day_sep) for i in range(len(avg_num_list))]
print(avg_num_list)



time_labels = pd.date_range(start="00:00", periods=len(avg_num_list), freq="15min").strftime('%H:%M')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 5))
plt.plot(time_labels, avg_num_list, marker='o', linestyle='-')
plt.title('车流量随时间变化（15分钟间隔）')
plt.xlabel('时间')
plt.ylabel('车流量')
plt.xticks(rotation=45, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(avg_num_list)
    score = silhouette_score(avg_num_list, labels)
    silhouette_scores.append(score)

# 可视化轮廓系数
plt.figure(figsize=(6, 4))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("不同聚类数量下的轮廓系数")
plt.xlabel("聚类数 k")
plt.ylabel("轮廓系数")
plt.grid(True)
plt.show()

# 选定最佳聚类数
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"最佳聚类数量为: {optimal_k}")

# 对数据聚类，查看每一天属于哪个类型
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
labels = kmeans.fit_predict(avg_num_list)

# 可视化：热力图 + 聚类标签
plt.figure(figsize=(12, 6))
sns.heatmap(avg_num_list, cmap='viridis')
plt.title(f"{day_sep}天车流量热力图（按聚类标签排序）")
plt.xlabel("15分钟时间段")
plt.ylabel("天数（按聚类）")
plt.show()

# 输出每类天数统计
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"类别 {u}: {c} 天")