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

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

year_start = pd.to_datetime(df['year'].astype(str) + '-01-01')
year_next = pd.to_datetime(df['year'].astype(str) + '-07-07')
df['progress'] = (df['date'] - year_start).dt.total_seconds() / (year_next - year_start).dt.total_seconds()

cov_df=pd.read_excel('./extra_data.xlsx')

df = df.merge(cov_df, on='year', how='left')

progress = df['progress'].values.reshape(-1,1)
poly = PolynomialFeatures(degree=2, include_bias=False)
P = poly.fit_transform(progress)

cov_names = ['ratio_gdp','num_student','num_laborer','ratio_urban']
C = df[cov_names].values

interacts = []
for j in range(C.shape[1]):
    col = C[:, j].reshape(-1,1)
    interacts.append(P * col)
interacts = np.hstack(interacts) if interacts else np.empty((len(df),0))

X = np.hstack([P, C, interacts])
y = df['rate'].values

alphas = np.logspace(-6, 6, 25)
model = RidgeCV(alphas=alphas, cv=5).fit(X, y)

future_year = 2025

ratio_gdp_2025 = 5
num_student_2025 = 670 
num_laborer_2025 = 8.7
ratio_urban_2025 = 66.3
cov_future = np.array([ratio_gdp_2025,num_student_2025,num_laborer_2025,ratio_urban_2025])

future_progress = np.linspace(0.05, 0.95, 10)
P_f = poly.transform(future_progress.reshape(-1,1))
C_f = np.tile(cov_future.reshape(1,-1), (len(future_progress),1))
inter_f = []
for j in range(C_f.shape[1]):
    inter_f.append(P_f * C_f[:, j].reshape(-1,1))
inter_f = np.hstack(inter_f) if inter_f else np.empty((len(future_progress),0))
X_f = np.hstack([P_f, C_f, inter_f])

y_f = model.predict(X_f)

start = pd.to_datetime(f"{future_year}-01-01")
end = pd.to_datetime(f"{future_year}-07-07")
future_dates = [start + (end - start) * p for p in future_progress]

out_df = pd.DataFrame({'date': future_dates, 'pred_rate': y_f})
print(out_df)
with open('./Q2_preds.md','w') as f:
    print(out_df,file=f)
plt.scatter(df['date'], df['rate'], alpha=0.5, label='history observation')
plt.plot(out_df['date'], out_df['pred_rate'], marker='o', color='red', label=f'{future_year} pred')
plt.legend()
plt.savefig('./Q2_fig.png')
# plt.show()