import pandas as pd, numpy as np, datetime, random, json
from scipy.optimize import curve_fit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def four_pl(x, A, K, k, x0):
    return A + (K - A) / (1 + np.exp(-k * (x - x0)))

def process_file(sheet_dict):
    result = pd.DataFrame(columns=['date','rate'])
    for year in sheet_dict:
        sh = sheet_dict[year]
        temp = sh.iloc[:, [0, -1]].copy()
        if year == '2018':
            for i in range(len(temp)-1):
                temp.iloc[i+1,0] = temp.iloc[i+1,0].date()
        else:
            if 1 in temp.index:
                try:
                    temp = temp.drop(index=1,axis=0).reset_index(drop=True)
                except Exception:
                    pass
            for i in range(len(temp)-1):
                s = str(temp.iloc[i+1,0])
                date_str = [year] + s.split('.')
                try:
                    date_list = [int(t) for t in date_str]
                    temp.iloc[i+1,0] = datetime.date(*date_list)
                except Exception:
                    temp.iloc[i+1,0] = pd.NaT
        temp.columns = ['date','rate']
        result = pd.concat([result, temp.iloc[1:,:].reset_index(drop=True)], axis=0)
    return result

xls = pd.ExcelFile('./Employment rate.xlsx')
sheets = {name: xls.parse(name) for name in xls.sheet_names}
df = process_file(sheets)

df['rate'] = pd.to_numeric(df['rate'].astype(str).str.strip().str.replace('%',''), errors='coerce')
if df['rate'].max() < 1.5:
    df['rate'] = df['rate'] * 100.0
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date','rate']).reset_index(drop=True)
df['year'] = df['date'].dt.year

year_start = pd.to_datetime(df['year'].astype(str) + '-01-01')
year_next = pd.to_datetime(df['year'].astype(str) + '-09-01')
df['progress'] = (df['date'] - year_start).dt.total_seconds() / (year_next - year_start).dt.total_seconds()
df['progress'] = np.clip(df['progress'], 0, 1)

cov_df = pd.read_excel('./extra_data.xlsx')
if 'year' not in cov_df.columns:
    raise ValueError("extra_data.xlsx must contain 'year' column")
df = df.merge(cov_df, on='year', how='left')

train_mask = df['year'] <= 2019
train_df = df[train_mask].copy().reset_index(drop=True)
progress_train = train_df['progress'].values
y_train = train_df['rate'].values

p0 = [np.min(y_train), np.max(y_train), 5.0, np.median(progress_train)]
lower = [0.0, 0.0, 0.01, 0.0]
upper = [200.0, 200.0, 50.0, 1.0]
params, pcov = curve_fit(four_pl, progress_train, y_train, p0=p0, bounds=(lower,upper), maxfev=10000)
A_fit, K_fit, k_fit, x0_fit = params
print("Fitted 4PL params:", params)

df['trend'] = four_pl(df['progress'].values, *params)
df['resid'] = df['rate'] - df['trend']

cov_names = [c for c in cov_df.columns if c!='year']
if len(cov_names)==0:
    raise ValueError("No covariates found in extra_data.xlsx (other than 'year').")

cov_scaler = StandardScaler()
C_all = cov_scaler.fit_transform(df[cov_names].values)
S_all = df['trend'].values.reshape(-1,1)
interacts_all = np.hstack([S_all * C_all[:,j].reshape(-1,1) for j in range(C_all.shape[1])]) if C_all.size else np.empty((len(df),0))
X_all = np.hstack([S_all, C_all, interacts_all])
y_resid = df['resid'].values

alphas = np.logspace(-6,6,25)
model_resid = RidgeCV(alphas=alphas, cv=3).fit(X_all, y_resid)
print("Residual model fitted. Coef sample:", model_resid.coef_[:8])

def interp_year(year, m=24):
    p_grid = np.linspace(0.0, 1.0, m)
    base = four_pl(p_grid, *params)
    row = cov_df[cov_df['year']==year]
    if row.shape[0]==0:
        row = cov_df.iloc[[-1]]
    cov_vec = row[cov_names].iloc[0].values.reshape(1,-1)
    cov_scaled = cov_scaler.transform(cov_vec)
    S_f = base.reshape(-1,1)
    C_f = np.tile(cov_scaled,(len(p_grid),1))
    inter_f = np.hstack([S_f * C_f[:,j].reshape(-1,1) for j in range(C_f.shape[1])]) if C_f.size else np.empty((len(p_grid),0))
    X_f = np.hstack([S_f, C_f, inter_f])
    resid_hat = model_resid.predict(X_f)
    y_hat = base + resid_hat
    start = pd.to_datetime(f"{year}-01-01")
    end = pd.to_datetime(f"{year}-09-01")
    dates = [(start + (end-start)*p).date() for p in p_grid]
    return pd.DataFrame({'year':year,'date':dates,'pred_rate':y_hat})

years = sorted(df['year'].unique())
interp_list = [interp_year(y, m=24) for y in years]
interp_all = pd.concat(interp_list, ignore_index=True)
interp_all.to_excel('./Attachment4_interpolated.xlsx', index=False)
print("Saved Attachment4_interpolated.xlsx:", len(interp_all), "rows")

def strict_mask_recover(year, hide_frac=0.3, n_iter=30):
    year_df = df[df['year']==year].reset_index()
    n = len(year_df)
    if n < 4:
        return None
    maes=[]
    for it in range(n_iter):
        hide_n = max(1,int(n*hide_frac))
        hide_idx = set(random.sample(range(n), hide_n))
        removed_global_idx = year_df.loc[list(hide_idx),'index'].values
        train_df2 = df.drop(index=removed_global_idx).reset_index(drop=True)
        try:
            params2, _ = curve_fit(four_pl, train_df2['progress'].values, train_df2['rate'].values,
                                   p0=p0, bounds=(lower,upper), maxfev=8000)
        except Exception:
            params2 = params
        train_df2['trend2'] = four_pl(train_df2['progress'].values, *params2)
        train_df2['resid2'] = train_df2['rate'] - train_df2['trend2']
        C2 = cov_scaler.transform(train_df2[cov_names].values)
        S2 = train_df2['trend2'].values.reshape(-1,1)
        X2 = np.hstack([S2, C2, np.hstack([S2 * C2[:,j].reshape(-1,1) for j in range(C2.shape[1])])])
        model2 = RidgeCV(alphas=alphas, cv=3).fit(X2, train_df2['resid2'].values)
        hidden_rows = year_df.loc[list(hide_idx)].copy()
        p_hidden = hidden_rows['progress'].values
        base_hidden = four_pl(p_hidden, *params2)
        cov_row = cov_df[cov_df['year']==year]
        if cov_row.shape[0]==0:
            cov_row = cov_df.iloc[[-1]]
        cov_scaled = cov_scaler.transform(cov_row[cov_names].values.reshape(1,-1))
        S_h = base_hidden.reshape(-1,1)
        C_h = np.tile(cov_scaled, (len(p_hidden),1))
        X_h = np.hstack([S_h, C_h, np.hstack([S_h * C_h[:,j].reshape(-1,1) for j in range(C_h.shape[1])])])
        resid_pred = model2.predict(X_h)
        y_pred = base_hidden + resid_pred
        y_true = hidden_rows['rate'].values
        maes.append(np.mean(np.abs(y_pred - y_true)))
    return np.mean(maes), np.std(maes)

years_for_eval = [min(years), int(np.median(years)), max(years)]
eval_results = {}
for y0 in years_for_eval:
    res = strict_mask_recover(y0, hide_frac=0.3, n_iter=30)
    eval_results[y0] = res
    print("Year", y0, "MAE(mean,std):", res)

report = {
    'fitted_params': {'A': float(A_fit), 'K': float(K_fit), 'k': float(k_fit), 'x0': float(x0_fit)},
    'mask_recover': {str(k): v for k, v in eval_results.items()}
}
with open('Attachment4_report.json','w') as f:
    json.dump(report, f, indent=2)
print("Saved Attachment4_report.json")
print("Done.")

def study_variable_impact(df, global_params, model_resid, cov_names, 
                         target_years=[2016, 2019, 2022],
                         n_iter=30):
    """
    研究变量数量对插值可信度的影响
    :param cov_names: 所有可用协变量名称列表
    :return: 结果字典 {变量数量: {年份: (可信度, MAE均值, MAE标准差)}}
    """
    results = {}
    max_vars = len(cov_names)

    # 为每个变量创建独立的 scaler（基于全量数据）
    var_scalers = {}
    for var in cov_names:
        scaler = StandardScaler()
        scaler.fit(df[[var]].values)
        var_scalers[var] = scaler

    for num_vars in range(0, max_vars + 1):
        year_results = {}
        print(f"正在评估 {num_vars} 个变量的情况...")

        for year in target_years:
            mae_list = []

            for _ in range(n_iter):
                # 1. 选择变量子集
                if num_vars == 0:
                    selected_vars = []
                else:
                    coef_values = model_resid.coef_[1:1+max_vars]
                    sorted_idx = np.argsort(-np.abs(coef_values))
                    selected_vars = [cov_names[i] for i in sorted_idx[:num_vars]]
                # 2. 降采样
                df_year = df[df['year'] == year].copy()
                # print(df_year)
                if len(df_year) < 4:
                    continue
                if 12 < len(df_year):
                    indices = sorted(random.sample(range(len(df_year)), 12))
                    lowres_df = df_year.iloc[indices].copy()
                else:
                    lowres_df = df_year.copy()

                # 3. 局部趋势
                params_local = global_params
                S_low = four_pl(lowres_df['progress'].values, *params_local).reshape(-1, 1)

                # 构造局部 X_low
                if selected_vars:
                    cov_scaled_list = []
                    for var in selected_vars:
                        cov_scaled_list.append(var_scalers[var].transform(lowres_df[[var]].values))
                    cov_low = np.hstack(cov_scaled_list)
                    X_low = np.hstack([S_low, cov_low])
                else:
                    X_low = S_low
                # print(X_low==S_low)

                y_resid_low = lowres_df['rate'].values - four_pl(lowres_df['progress'].values, *params_local)
                model_local = Ridge(alpha=model_resid.alpha_).fit(X_low, y_resid_low)

                # 4. 高分辨率局部预测
                S_high = four_pl(REF_GRID, *params_local).reshape(-1, 1)
                if selected_vars:
                    cov_values = []
                    for var in selected_vars:
                        year_value = cov_df[cov_df['year'] == year][var].values[0]
                        cov_values.append(var_scalers[var].transform([[year_value]])[0][0])
                    cov_high = np.tile(cov_values, (len(REF_GRID), 1))
                    X_high = np.hstack([S_high, cov_high])
                else:
                    X_high = S_high
                resid_pred = model_local.predict(X_high)
                y_pred = S_high.flatten() + resid_pred

                # 5. 高分辨率参考预测（用相同变量子集）
                S_ref = four_pl(REF_GRID, *global_params).reshape(-1, 1)
                if selected_vars:
                    cov_ref_values = []
                    for var in selected_vars:
                        year_value = cov_df[cov_df['year'] == year][var].values[0]
                        cov_ref_values.append(var_scalers[var].transform([[year_value]])[0][0])
                    cov_ref = np.tile(cov_ref_values, (len(REF_GRID), 1))
                    X_ref = np.hstack([S_ref, cov_ref])
                else:
                    X_ref = S_ref

                # 用全量数据拟合参考残差模型
                if selected_vars:
                    cov_all = np.hstack([var_scalers[var].transform(df[[var]].values) for var in selected_vars])
                    X_all = np.hstack([four_pl(df['progress'].values, *global_params).reshape(-1, 1), cov_all])
                else:
                    X_all = four_pl(df['progress'].values, *global_params).reshape(-1, 1)
                y_resid_all = df['rate'].values - four_pl(df['progress'].values, *global_params)
                model_ref = Ridge(alpha=model_resid.alpha_).fit(X_all, y_resid_all)

                resid_ref = model_ref.predict(X_ref)
                y_ref = S_ref.flatten() + resid_ref

                # 6. 计算 MAE
                mae = np.mean(np.abs(y_pred - y_ref))
                mae_list.append(mae)

            mean_mae = np.mean(mae_list)
            std_mae = np.std(mae_list)
            confidence = 1 / (mean_mae + std_mae + 1e-5)

            year_results[year] = {
                "confidence": float(confidence),
                "mae_mean": float(mean_mae),
                "mae_std": float(std_mae)
            }

        results[num_vars] = year_results
    return results


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from scipy.optimize import curve_fit
    import random
    import matplotlib
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 高分辨率参考网格（固定为24点/年）
    REF_GRID = np.linspace(0, 1, 24)
    
    # 分辨率参数设置
    resolutions = [5, 10, 15, 20, 24]
    target_years = list(range(2018,2023,1))
    n_iter = 50  # 蒙特卡洛迭代次数
    
    confidence_records = {year: [] for year in target_years}
    resolution_results = {}
    
    # 获取全局模型参数
    global_params = [A_fit, K_fit, k_fit, x0_fit]
    
    # 执行变量数量研究
    var_impact_results = study_variable_impact(
        df, global_params, model_resid, cov_names,
        target_years=target_years, n_iter=n_iter
    )
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 准备绘图数据
    num_vars_list = sorted(var_impact_results.keys())
    confidence_data = {year: [] for year in target_years}
    
    for num_vars in num_vars_list:
        for year in target_years:
            if year in var_impact_results[num_vars]:
                confidence_data[year].append(var_impact_results[num_vars][year]['confidence'])
    
    # 绘制曲线
    for year in target_years:
        plt.plot(num_vars_list, confidence_data[year], 
                 marker='o', linestyle='-', linewidth=2.5,
                 label=f"{year}年")
    
    # 图表装饰
    plt.xlabel('使用的变量数量')
    plt.ylabel('插值可信度 (1/(MAE_mean+MAE_std))')
    plt.title('变量数量与插值可信度关系')
    plt.xticks(num_vars_list)
    plt.grid(True, alpha=0.3)
    plt.legend(title='年份')
    
    # 添加最优值标记
    for year in target_years:
        conf_values = confidence_data[year]
        max_idx = np.argmax(conf_values)
    
    plt.tight_layout()
    plt.savefig('variable_count_vs_confidence.png', dpi=300)
    plt.close()
    
    # 保存结果
    report['variable_impact'] = var_impact_results
    with open('variable_impact_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("变量数量研究完成，结果已保存")
    
    
    
    for m in resolutions:
        resolution_results[m] = {}
        for year in target_years:
            df_year = df[df["year"] == year].copy()
            if len(df_year) < 4:  # 确保足够拟合点
                continue
                
            mae_list = []
            for _ in range(n_iter):
                # 1. 降采样到m个点
                if m < len(df_year):
                    indices = sorted(random.sample(range(len(df_year)), m))
                    lowres_df = df_year.iloc[indices].copy()
                else:
                    lowres_df = df_year.copy()
                
                # 2. 训练局部趋势模型
                try:
                    params_local, _ = curve_fit(four_pl, 
                                               lowres_df['progress'].values,
                                               lowres_df['rate'].values,
                                               p0=global_params,
                                               bounds=(lower, upper))
                except RuntimeError:
                    params_local = global_params
                
                # 3. 在低分辨率点上训练残差模型
                S_low = four_pl(lowres_df['progress'].values, *params_local).reshape(-1, 1)
                cov_low = cov_scaler.transform(lowres_df[cov_names].values)
                inter_low = np.hstack([S_low * cov_low[:, j].reshape(-1,1) for j in range(cov_low.shape[1])])
                X_low = np.hstack([S_low, cov_low, inter_low])
                y_resid_low = lowres_df['rate'].values - four_pl(lowres_df['progress'].values, *params_local)
                model_local = Ridge(alpha=model_resid.alpha_).fit(X_low, y_resid_low)
                
                # 4. 在高分辨率网格上插值
                S_high = four_pl(REF_GRID, *params_local).reshape(-1, 1)
                cov_high = cov_scaler.transform(lowres_df[cov_names].iloc[0].values.reshape(1, -1))
                cov_high = np.tile(cov_high, (len(REF_GRID), 1))
                inter_high = np.hstack([S_high * cov_high[:, j].reshape(-1,1) for j in range(cov_high.shape[1])])
                X_high = np.hstack([S_high, cov_high, inter_high])
                resid_pred = model_local.predict(X_high)
                y_pred = S_high.flatten() + resid_pred
                
                # 5. 获取参考值（全局模型）
                S_ref = four_pl(REF_GRID, *global_params).reshape(-1, 1)
                cov_ref = cov_scaler.transform(cov_df[cov_df['year']==year][cov_names].values)
                cov_ref = np.tile(cov_ref, (len(REF_GRID), 1))
                inter_ref = np.hstack([S_ref * cov_ref[:, j].reshape(-1,1) for j in range(cov_ref.shape[1])])
                X_ref = np.hstack([S_ref, cov_ref, inter_ref])
                resid_ref = model_resid.predict(X_ref)
                y_ref = S_ref.flatten() + resid_ref
                
                # 6. 计算MAE
                mae = np.mean(np.abs(y_pred - y_ref))
                mae_list.append(mae)
            
            # 计算统计量
            mean_mae = np.mean(mae_list)
            std_mae = np.std(mae_list)
            confidence = 1 / (mean_mae + std_mae + 1e-5)
            
            resolution_results[m][year] = {
                "mae_mean": float(mean_mae),
                "mae_std": float(std_mae),
                "confidence": float(confidence)
            }
            confidence_records[year].append(confidence)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    for year in target_years:
        plt.plot(resolutions, confidence_records[year], 'o-', label=f"{year}")
    plt.xlabel("分辨率")
    plt.ylabel("插值可信度 (1/(MAE_mean+MAE_std))")
    plt.title("分辨率与插值可信度关系")
    plt.legend()
    plt.grid(True)
    plt.savefig("Resolution vs Confidence.png", dpi=300)
    plt.close()
    
    # 保存结果
    report["resolution_analysis"] = resolution_results
    with open("resolution_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Saved corrected resolution_vs_confidence.png and resolution_analysis.json")