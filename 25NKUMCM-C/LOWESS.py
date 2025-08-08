import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

df=pd.read_excel(r'./Employment rate.xlsx',sheet_name=None)
# print(df['2021'])
def process(df):
    result=pd.DataFrame(columns=['date','rate'])
    for year in df:
        sh=df[year]  
        temp=sh.iloc[:,[0,-1]].copy()
        if year=='2018':
            for i in range(len(temp)-1):
                temp.iloc[i+1,0]=temp.iloc[i+1,0].date()
        else:
            for i in range(len(temp)-1):
                s=str(temp.iloc[i+1,0])
                date_str=[year]+s.split('.')
                date_list=[int(t) for t in date_str]
                temp.iloc[i+1,0]=datetime.date(*date_list)
        temp.columns=['date','rate']
        result=pd.concat([result,temp.iloc[1:,:].reset_index(drop=True)],axis=0)
    return result

df=process(df)
df['date']=pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
yearly_curves = {}
yearly_avg = []

plt.figure(figsize=(10,6))

for year, group in df.groupby('year'):
    group['time'] = group['date'].map(datetime.datetime.toordinal)
    lowess_res = sm.nonparametric.lowess(group['rate'], group['time'], frac=0.3)

    t_fit, y_fit = lowess_res[:, 0], lowess_res[:, 1]
    dates_fit = [datetime.datetime.fromordinal(int(t)) for t in t_fit]

    plt.scatter(group['date'], group['rate'], alpha=0.5, label=f"{year} raw data")
    plt.plot(dates_fit, y_fit, linewidth=2, label=f"{year} LOWESS")

plt.title("raw data vs LOWESS ")
plt.xlabel("time")
plt.ylabel("rate")
plt.legend()
plt.grid(True)
plt.show()