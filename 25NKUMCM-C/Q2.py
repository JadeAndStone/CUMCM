# import matplotlib.pyplot as plt
# import pandas as pd
# import datetime
# import numpy as np
# from sklearn.linear_model import RidgeCV
# from sklearn.preprocessing import StandardScaler
# from scipy.optimize import curve_fit

# # Sigmoid 函数定义
# def sigmoid(x, L, x0, k):
#     return L / (1 + np.exp(-k * (x - x0)))

# # 数据处理部分保持不变
# df = pd.read_excel('./Employment rate.xlsx', sheet_name=None)

# def process(df):
#     result = pd.DataFrame(columns=['date', 'rate'])
#     for year in df:
#         sh = df[year]  
#         temp = sh.iloc[:, [0, -1]].copy()
#         if year == '2018':
#             for i in range(len(temp) - 1):
#                 temp.iloc[i + 1, 0] = temp.iloc[i + 1, 0].date()
#         else:
#             for i in range(len(temp) - 1):
#                 s = str(temp.iloc[i + 1, 0])
#                 date_str = [year] + s.split('.')
#                 date_list = [int(t) for t in date_str]
#                 temp.iloc[i + 1, 0] = datetime.date(*date_list)
#         temp.columns = ['date', 'rate']
#         result = pd.concat([result, temp.iloc[1:, :].reset_index(drop=True)], axis=0)
#     return result

# df = process(df)

# # 确保rate列为数值类型 - 修正关键点
# df['rate'] = pd.to_numeric(df['rate'].astype(str).str.strip().str.replace('%', ''), errors='coerce')

# df['date'] = pd.to_datetime(df['date'])
# df['year'] = df['date'].dt.year

# year_start = pd.to_datetime(df['year'].astype(str) + '-01-16')
# year_next = pd.to_datetime(df['year'].astype(str) + '-07-07')
# df['progress'] = (df['date'] - year_start).dt.total_seconds() / (year_next - year_start).dt.total_seconds()

# cov_df = pd.read_excel('./extra_data.xlsx')
# df = df.merge(cov_df, on='year', how='left')
# progress = df['progress'].values
# y = df['rate'].values

# # 用 sigmoid 拟合代替多项式特征
# # 拟合全局 sigmoid 曲线获取基础参数
# if df['rate'].dtype != float:
#     df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    
# # 移除可能的NaN值
# valid_idx = ~np.isnan(y)
# progress = progress[valid_idx]
# y = y[valid_idx]

# p0 = [np.max(y), np.median(progress), 1]  # 初始参数猜测
# params, _ = curve_fit(sigmoid, progress, y, p0=p0)
# L, x0, k = params

# # 生成 sigmoid 基础特征
# S = sigmoid(progress, L, x0, k).reshape(-1, 1)

# # 协变量处理
# cov_names = ['ratio_gdp', 'num_student', 'num_laborer', 'ratio_urban']
# cov_scaler = StandardScaler()
# C = cov_scaler.fit_transform(df[cov_names].values)

# # 交互特征：sigmoid特征与每个协变量的交互
# interacts = []
# for j in range(C.shape[1]):
#     interacts.append(S * C[:, j].reshape(-1, 1))
# interacts = np.hstack(interacts) if interacts else np.empty((len(df), 0))

# # 构建最终特征矩阵
# X = np.hstack([S, C, interacts])

# # 使用岭回归拟合（保持正则化）
# alphas = np.logspace(-6, 6, 25)
# model = RidgeCV(alphas=alphas, cv=4).fit(X, y)

# # 预测2022年数据
# future_year = 2022
# ratio_gdp_2025 = 3.1
# num_student_2025 = 704.2
# num_laborer_2025 = 8.76
# ratio_urban_2025 = 65.22

# # 处理未来协变量
# cov_future = cov_scaler.transform(np.array([
#     ratio_gdp_2025,
#     num_student_2025,
#     num_laborer_2025,
#     ratio_urban_2025
# ]).reshape(1, -1))

# # 生成预测时间点
# future_progress = np.linspace(0.05, 0.95, 10)
# S_f = sigmoid(future_progress, L, x0, k).reshape(-1, 1)

# # 复制协变量到每个时间点
# C_f = np.tile(cov_future, (len(future_progress), 1))

# # 构建交互特征
# inter_f = []
# for j in range(C_f.shape[1]):
#     inter_f.append(S_f * C_f[:, j].reshape(-1, 1))
# inter_f = np.hstack(inter_f) if inter_f else np.empty((len(future_progress), 0))

# # 构建预测特征矩阵
# X_f = np.hstack([S_f, C_f, inter_f])

# # 预测就业率
# y_f = model.predict(X_f)

# # 生成预测日期并输出结果
# start = pd.to_datetime(f"{future_year}-01-16")
# end = pd.to_datetime(f"{future_year}-07-07")
# future_dates = [start + (end - start) * p for p in future_progress]

# out_df = pd.DataFrame({'date': future_dates, 'pred_rate': y_f})
# print(out_df)

# # # 保存预测结果
# # with open('./Q2_preds.md', 'w') as f:
# #     print(out_df.to_markdown(index=False), file=f)

# # 可视化结果
# plt.figure(figsize=(10, 6))
# plt.scatter(df['date'], df['rate'], alpha=0.5, label='Historical Data')
# plt.plot(out_df['date'], out_df['pred_rate'], 'r-o', label=f'{future_year} Prediction')
# plt.xlabel('Date')
# plt.ylabel('Employment Rate')
# plt.title('Employment Rate Prediction')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('./Q2_fig.png')
# plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd
# import datetime
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import matplotlib
# from scipy.optimize import curve_fit
# from matplotlib.lines import Line2D

# # 设置中文字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

# # 四参数逻辑函数定义 (4PL)
# def four_pl(x, A, K, k, x0):
#     """四参数逻辑函数"""
#     return A + (K - A) / (1 + np.exp(-k * (x - x0)))

# # 数据处理函数
# def process(df):
#     result = pd.DataFrame(columns=['date', 'rate'])
#     for year in df:
#         sh = df[year]  
#         temp = sh.iloc[:, [0, -1]].copy()
#         if year == '2018':
#             for i in range(len(temp) - 1):
#                 temp.iloc[i + 1, 0] = temp.iloc[i + 1, 0].date()
#         else:
#             if year == '2020':
#                 temp = temp.drop(index=1,axis=0).reset_index(drop=True)
#             for i in range(len(temp) - 1):
#                 s = str(temp.iloc[i + 1, 0])
#                 date_str = [year] + s.split('.')
#                 date_list = [int(t) for t in date_str]
#                 temp.iloc[i + 1, 0] = datetime.date(*date_list)
#         temp.columns = ['date', 'rate']
#         result = pd.concat([result, temp.iloc[1:, :].reset_index(drop=True)], axis=0)
#     return result

# # 读取就业率数据
# df = pd.read_excel('./Employment rate.xlsx', sheet_name=None)
# df = process(df)

# # 确保rate列为百分比值
# df['rate'] = pd.to_numeric(df['rate'].astype(str).str.strip().str.replace('%', ''), errors='coerce')
# df = df.dropna(subset=['rate'])

# # 检查数据范围 - 关键修复
# if df['rate'].max() < 1.5:
#     print("警告: 就业率数据低于1.5，可能是小数形式，已转换为百分比值")
#     df['rate'] = df['rate'] * 100

# print(f"就业率统计: 最小值={df['rate'].min()}%, 最大值={df['rate'].max()}%")

# # 处理日期和计算时间进度
# df['date'] = pd.to_datetime(df['date'])
# df['year'] = df['date'].dt.year

# # 创建分组标签
# df['group'] = df['year'].apply(
#     lambda y: "疫情前 (2016-2019)" if y <= 2019 else "疫情期间 (2020-2022)"
# )

# year_start = pd.to_datetime(df['year'].astype(str) + '-01-16')
# year_next = pd.to_datetime(df['year'].astype(str) + '-07-07')
# df['progress'] = (df['date'] - year_start).dt.total_seconds() / (year_next - year_start).dt.total_seconds()

# # 保证进度在0-1范围内
# df['progress'] = np.clip(df['progress'], 0, 1)

# # ============================
# # 按年份独立拟合（恢复此部分）
# # ============================
# print("\n按年份独立拟合结果:")
# years = sorted(df['year'].unique())
# params_by_year = {}

# plt.figure(figsize=(15, 10))
# plt.suptitle('按年份独立拟合就业率趋势', fontsize=16)

# for idx, year in enumerate(years):
#     year_data = df[df['year'] == year].copy()
#     if len(year_data) < 5:
#         print(f"跳过{year}年，数据点不足")
#         continue
        
#     x_data = year_data['progress'].values
#     y_data = year_data['rate'].values
    
#     # 初始参数估计
#     A_est = y_data.min()
#     K_est = y_data.max()
#     x0_est = np.median(x_data)
#     k_est = 5.0
    
#     # 设置合理的参数边界
#     lower_bounds = [A_est - 15, K_est - 15, 0.1, 0]
#     upper_bounds = [A_est + 15, K_est + 15, 15, 1]
    
#     try:
#         params, _ = curve_fit(
#             four_pl,
#             x_data,
#             y_data,
#             p0=[A_est, K_est, k_est, x0_est],
#             bounds=(lower_bounds, upper_bounds),
#             maxfev=10000
#         )
#         params_by_year[year] = params
        
#         # 绘制子图
#         plt.subplot(3, 3, idx+1)
#         plt.scatter(x_data, y_data, label=f'{year}年数据')
        
#         x_fit = np.linspace(0, 1, 100)
#         y_fit = four_pl(x_fit, *params)
#         plt.plot(x_fit, y_fit, 'r-', label='拟合曲线')
        
#         plt.title(f'{year}年就业率拟合')
#         plt.xlabel('时间进度')
#         plt.ylabel('就业率 (%)')
#         plt.legend()
#         plt.grid(True)
        
#         # 计算拟合质量指标
#         residuals = y_data - four_pl(x_data, *params)
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y_data - np.mean(y_data))**2)
#         r_squared = 1 - (ss_res / ss_tot)
        
#         print(f"{year}年拟合结果: A={params[0]:.2f}, K={params[1]:.2f}, k={params[2]:.4f}, x0={params[3]:.4f}, R²={r_squared:.3f}")
        
#     except Exception as e:
#         print(f"无法拟合{year}年: {e}")
#         params_by_year[year] = None
#         # 只绘制散点图
#         plt.subplot(3, 3, idx+1)
#         plt.scatter(x_data, y_data, label=f'{year}年数据')
#         plt.title(f'{year}年数据分布')
#         plt.xlabel('时间进度')
#         plt.ylabel('就业率 (%)')
#         plt.legend()
#         plt.grid(True)

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig('./yearly_fitting.png')
# plt.close()

# # ============================
# # 按疫情分组拟合
# # ============================
# print("\n按疫情分组拟合结果:")
# group_params = {}
# group_ci = {}
# groups = ["疫情前 (2016-2019)", "疫情期间 (2020-2022)"]

# plt.figure(figsize=(14, 6))

# for group_idx, group in enumerate(groups):
#     group_data = df[df['group'] == group].copy()
#     group_data['progress'] = np.clip(group_data['progress'], 0, 1)
    
#     # 排除极端异常值
#     q_low = group_data['rate'].quantile(0.01)
#     q_high = group_data['rate'].quantile(0.99)
#     group_data = group_data[(group_data['rate'] >= q_low) & (group_data['rate'] <= q_high)]
    
#     if len(group_data) < 10:
#         print(f"{group}组数据点不足，跳过拟合")
#         continue
        
#     x_data = group_data['progress'].values
#     y_data = group_data['rate'].values
    
#     # 初始参数估计
#     A_est = np.percentile(y_data, 5)
#     K_est = np.percentile(y_data, 95)
#     x0_est = np.median(x_data)
#     k_est = 5.0 if group == "疫情前 (2016-2019)" else 2.0
    
#     # 设置合理的参数边界
#     lower_bounds = [A_est - 10, K_est - 10, 0.1, 0]
#     upper_bounds = [A_est + 10, K_est + 10, 15, 1]
    
#     try:
#         params, pcov = curve_fit(
#             four_pl,
#             x_data,
#             y_data,
#             p0=[A_est, K_est, k_est, x0_est],
#             bounds=(lower_bounds, upper_bounds),
#             maxfev=10000
#         )
#         group_params[group] = params
        
#         # 计算拟合质量指标
#         residuals = y_data - four_pl(x_data, *params)
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y_data - np.mean(y_data))**2)
#         r_squared = 1 - (ss_res / ss_tot)
        
#         # 可视化
#         plt.subplot(1, len(groups), group_idx+1)
#         plt.scatter(x_data, y_data, alpha=0.6, label=f"{group}数据点")
        
#         x_fit = np.linspace(0, 1, 100)
#         y_fit = four_pl(x_fit, *params)
#         plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f"{group}拟合曲线")
        
#         plt.title(f"{group}就业率变化拟合 (R²={r_squared:.3f})")
#         plt.xlabel("时间进度 (0-1)")
#         plt.ylabel("就业率 (%)")
#         plt.legend()
#         plt.grid(True)
        
#         print(f"{group}拟合结果: A={params[0]:.2f}, K={params[1]:.2f}, k={params[2]:.4f}, x0={params[3]:.4f}, R²={r_squared:.3f}")
        
#         # 计算95%置信区间
#         n_simulations = 1000
#         y_sims = np.zeros((len(x_fit), n_simulations))
        
#         for i in range(n_simulations):
#             random_params = np.random.multivariate_normal(params, pcov)
#             y_sims[:, i] = four_pl(x_fit, *random_params)
        
#         ci_lower = np.percentile(y_sims, 2.5, axis=1)
#         ci_upper = np.percentile(y_sims, 97.5, axis=1)
        
#         plt.fill_between(x_fit, ci_lower, ci_upper, color='r', alpha=0.2, label='95%置信区间')
        
#     except Exception as e:
#         print(f"无法拟合 {group} 组: {e}")
#         group_params[group] = None
#         group_ci[group] = None
#         # 只绘制散点图
#         plt.subplot(1, len(groups), group_idx+1)
#         plt.scatter(x_data, y_data, alpha=0.6, label=f"{group}数据点")
#         plt.title(f"{group}数据分布")
#         plt.xlabel("时间进度 (0-1)")
#         plt.ylabel("就业率 (%)")
#         plt.legend()
#         plt.grid(True)

# plt.tight_layout()
# plt.savefig('./Q2_covid_analysis.png')
# plt.close()

# # ============================
# # 分组拟合结果分析
# # ============================
# if len(group_params) == 2:
#     pre_covid_params = group_params.get("疫情前 (2016-2019)")
#     covid_params = group_params.get("疫情期间 (2020-2022)")
    
#     if pre_covid_params is not None and covid_params is not None:
#         print("\n分组拟合参数对比分析:")
#         print(f"{'参数':<8} {'疫情前':<15} {'疫情期间':<15} {'变化率(%)':<10}")
#         print("-" * 45)
        
#         def calc_change(new, old):
#             return ((new - old) / old) * 100 if old != 0 else float('nan')
        
#         param_names = ["A (初期)", "K (末期)", "k (陡度)", "x₀ (拐点)"]
        
#         for i, name in enumerate(param_names):
#             pre_val = pre_covid_params[i]
#             covid_val = covid_params[i]
#             change = calc_change(covid_val, pre_val)
#             print(f"{name:<8} {pre_val:.3f}{'':<10} {covid_val:.3f}{'':<10} {change:.2f}%")
        
#         # 特别分析趋势变化
#         print("\n疫情对就业趋势的影响分析:")
#         print("1. 就业起点变化 (A):", 
#               "上升" if covid_val > pre_val else "下降", 
#               f"({change:.2f}%)")
        
#         print("2. 就业终点变化 (K):", 
#               "上升" if covid_val > pre_val else "下降", 
#               f"({change:.2f}%)")
        
#         print("3. 就业变化陡度 (k):", 
#               "变陡" if covid_val > pre_val else "变缓", 
#               f"({change:.2f}%)")
        
#         print("4. 拐点位置变化 (x₀):", 
#               "推后" if covid_val > pre_val else "提前", 
#               f"({change:.2f}%)")

#         # 计算模型差异
#         x_eval = np.linspace(0, 1, 10)
#         pre_pred = four_pl(x_eval, *pre_covid_params)
#         covid_pred = four_pl(x_eval, *covid_params)
#         avg_change = np.mean(covid_pred - pre_pred)
#         max_change = np.max(np.abs(covid_pred - pre_pred))
        
#         print(f"\n综合影响: 平均就业率变化: {avg_change:.2f}%")
#         print(f"最大变化幅度: {max_change:.2f}%")
# else:
#     print("\n警告: 未获得完整分组拟合结果，跳过分析")
    

# # ============================
# # 各年份拟合曲线对比图（关键新增功能）
# # ============================
# plt.figure(figsize=(12, 8))

# # 创建颜色映射 (2016-2022)
# years = sorted(df['year'].unique())
# color_map = plt.get_cmap('viridis')
# year_colors = {year: color_map(i/(len(years)-1)) for i, year in enumerate(years)}

# # 创建图例元素
# legend_elements = []

# # 绘制所有年份拟合曲线
# for year in years:
#     if year in params_by_year and params_by_year[year] is not None:
#         # 获取拟合参数
#         A, K, k, x0 = params_by_year[year]
        
#         # 生成拟合曲线
#         x_range = np.linspace(0, 1, 100)
#         y_fit = four_pl(x_range, A, K, k, x0)
        
#         # 绘制曲线
#         plt.plot(
#             x_range, 
#             y_fit, 
#             color=year_colors[year], 
#             linewidth=3 if year > 2019 else 2,
#             linestyle='-' if year <= 2019 else '--'
#         )
        
#         # 标记特殊年份
#         if year == 2020:
#             plt.text(
#                 x_range[10], 
#                 y_fit[10], 
#                 '2020(疫情开始)', 
#                 fontsize=10,
#                 color=year_colors[year],
#                 bbox=dict(facecolor='white', alpha=0.7)
#             )
        
#         # 添加图例元素
#         legend_elements.append(Line2D(
#             [0], [0], 
#             color=year_colors[year], 
#             lw=2, 
#             label=f'{year}年拟合曲线'
#         ))

# # 添加实际数据点
# for year in years:
#     year_data = df[df['year'] == year]
#     plt.scatter(
#         year_data['progress'],
#         year_data['rate'],
#         color=year_colors[year],
#         alpha=0.8,
#         s=60,
#         marker='o' if year <= 2019 else 's'
#     )

# # 添加疫情时间线
# plt.axvline(x=0.8, color='gray', linestyle=':', linewidth=2)
# plt.text(0.81, plt.ylim()[1]*0.95, '疫情开始(2020)', fontsize=12, color='gray')

# # 设置图表属性
# plt.title('2016-2022年就业率趋势变化对比', fontsize=16)
# plt.xlabel('时间进度 (0-1)')
# plt.ylabel('就业率 (%)')
# plt.grid(True, linestyle='--', alpha=0.7)

# # 创建复合图例
# legend_elements += [
#     Line2D([0], [0], marker='o', color='w', label='2016-2019年实际数据点',
#            markerfacecolor='gray', markersize=10),
#     Line2D([0], [0], marker='s', color='w', label='2020-2022年实际数据点',
#            markerfacecolor='gray', markersize=10),
#     Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='疫情时间线')
# ]

# plt.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

# # 突出关键区域
# plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]], x1=0.7, x2=0.9, 
#                   color='lightpink', alpha=0.2)
# plt.text(0.72, plt.ylim()[0]+1, '关键转折区域', fontsize=10, color='darkred')

# # 保存对比图
# plt.tight_layout()
# plt.savefig('./Q2_yearly_comparison.png', dpi=300)
# plt.close()

# # ============================
# # 新增：疫情前后拟合曲线对比（同图叠加）
# # ============================
# plt.figure(figsize=(10, 6))

# # 为每个分组设定颜色
# group_colors = {
#     "疫情前 (2016-2019)": "blue",
#     "疫情期间 (2020-2022)": "red"
# }

# # 遍历每个分组
# for group in groups:
#     if group in group_params and group_params[group] is not None:
#         params = group_params[group]
#         group_data = df[df['group'] == group]
        
#         # 绘制数据点
#         plt.scatter(
#             group_data['progress'],
#             group_data['rate'],
#             color=group_colors[group],
#             alpha=0.5,
#             label=f'{group}数据点'
#         )
        
#         # 绘制拟合曲线
#         x_fit = np.linspace(0, 1, 100)
#         y_fit = four_pl(x_fit, *params)
#         plt.plot(
#             x_fit,
#             y_fit,
#             color=group_colors[group],
#             linewidth=2.5,
#             label=f'{group}拟合曲线'
#         )
        
#         # 绘制置信区间
#         if group in group_ci and group_ci[group] is not None:
#             ci_lower, ci_upper = group_ci[group]
#             plt.fill_between(
#                 x_fit,
#                 ci_lower,
#                 ci_upper,
#                 color=group_colors[group],
#                 alpha=0.2
#             )

# plt.title('疫情前后就业率变化趋势对比')
# plt.xlabel('时间进度 (0-1)')
# plt.ylabel('就业率 (%)')
# plt.legend(loc='best')
# plt.grid(True)

# # 添加关键参数标记
# if len(group_params) == 2:
#     pre_params = group_params["疫情前 (2016-2019)"]
#     covid_params = group_params["疫情期间 (2020-2022)"]
    
#     # 添加拐点标记
#     plt.scatter(pre_params[3], four_pl(pre_params[3], *pre_params), 
#                s=100, marker='o', color='blue', edgecolors='white', 
#                zorder=10, label='疫情前拐点')
#     plt.scatter(covid_params[3], four_pl(covid_params[3], *covid_params), 
#                s=100, marker='o', color='red', edgecolors='white', 
#                zorder=10, label='疫情期间拐点')
    
#     # 添加趋势线说明
#     plt.annotate(f'起始差: +{(covid_params[0]-pre_params[0]):.1f}%\n结束差: -{(pre_params[1]-covid_params[1]):.1f}%',
#                 xy=(0.05, 0.85), xycoords='axes fraction', 
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# plt.tight_layout()
# plt.savefig('./Q2_covid_curve_comparison.png')
# plt.close()

import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.optimize import curve_fit
import matplotlib
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 四参数逻辑函数定义 (4PL)
def four_pl(x, A, K, k, x0):
    """四参数逻辑函数"""
    return A + (K - A) / (1 + np.exp(-k * (x - x0)))

# Sigmoid 函数定义
def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

# 数据处理函数
def process(df):
    result = pd.DataFrame(columns=['date', 'rate'])
    for year in df:
        sh = df[year]  
        temp = sh.iloc[:, [0, -1]].copy()
        if year == '2018':
            for i in range(len(temp) - 1):
                temp.iloc[i + 1, 0] = temp.iloc[i + 1, 0].date()
        else:
            if year == '2020':
                temp = temp.drop(index=1,axis=0).reset_index(drop=True)
            for i in range(len(temp) - 1):
                s = str(temp.iloc[i + 1, 0])
                date_str = [year] + s.split('.')
                date_list = [int(t) for t in date_str]
                temp.iloc[i + 1, 0] = datetime.date(*date_list)
        temp.columns = ['date', 'rate']
        result = pd.concat([result, temp.iloc[1:, :].reset_index(drop=True)], axis=0)
    return result

# 读取就业率数据
df = pd.read_excel('./Employment rate.xlsx', sheet_name=None)
df = process(df)

# 确保rate列为数值类型
df['rate'] = pd.to_numeric(df['rate'].astype(str).str.strip().str.replace('%', ''), errors='coerce')

# 检查数据范围 - 关键修复
if df['rate'].max() < 1.5:
    print("警告: 就业率数据低于1.5，可能是小数形式，已转换为百分比值")
    df['rate'] = df['rate'] * 100

print(f"就业率统计: 最小值={df['rate'].min()}%, 最大值={df['rate'].max()}%")

# 处理日期和计算时间进度
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# 创建分组标签
df['group'] = df['year'].apply(
    lambda y: "疫情前 (2016-2019)" if y <= 2019 else "疫情期间 (2020-2022)"
)

year_start = pd.to_datetime(df['year'].astype(str) + '-01-01')
year_next = pd.to_datetime(df['year'].astype(str) + '-08-01')
df['progress'] = (df['date'] - year_start).dt.total_seconds() / (year_next - year_start).dt.total_seconds()

# 保证进度在0-1范围内
df['progress'] = np.clip(df['progress'], 0, 1)

# ============================
# 协变量分析部分（恢复）
# ============================
print("\n=== 协变量分析部分 ===")

# 读取协变量数据
cov_df = pd.read_excel('./extra_data.xlsx')
df = df.merge(cov_df, on='year', how='left')
progress = df['progress'].values
y = df['rate'].values

# 移除可能的NaN值
valid_idx = ~np.isnan(y)
progress = progress[valid_idx]
y = y[valid_idx]

# 拟合全局 sigmoid 曲线获取基础参数 - 关键修复
print("拟合全局 sigmoid 曲线...")

p0 = [np.max(y), np.median(progress), 1]  # 初始参数猜测
params, _ = curve_fit(sigmoid, progress, y, p0=p0, maxfev=2000)  # 增加评估次数
print("标准拟合成功")

# 提取参数
L, x0, k = params
print(f"拟合参数: L={L:.2f}, x0={x0:.2f}, k={k:.2f}")

# 生成 sigmoid 基础特征
S = sigmoid(progress, L, x0, k).reshape(-1, 1)

# 协变量处理
cov_names = ['ratio_gdp', 'num_student', 'num_laborer', 'ratio_urban']
cov_scaler = StandardScaler()
C = cov_scaler.fit_transform(df[cov_names].values)

# 交互特征：sigmoid特征与每个协变量的交互
interacts = []
for j in range(C.shape[1]):
    interacts.append(S * C[:, j].reshape(-1, 1))
interacts = np.hstack(interacts) if interacts else np.empty((len(df), 0))

# 构建最终特征矩阵
X = np.hstack([S, C, interacts])

# 使用岭回归拟合（保持正则化）
alphas = np.logspace(-6, 6, 25)
model = RidgeCV(alphas=alphas, cv=3).fit(X, y)

# ============================
# 预测2025和2026年数据
# ============================
print("\n=== 预测2025和2026年数据 ===")

# 从Excel文件读取预测数据
predict_data = pd.read_excel('./predict_data.xlsx')
predict_data = predict_data.drop(index=2,axis=0)

# 提取预测数据
future_years = [2025, 2026]
all_predictions = pd.DataFrame()

for _, row in predict_data.iterrows():
    future_year = int(row['year'])
    ratio_gdp = row['ratio_gdp']
    num_student = row['num_student']
    num_laborer = row['num_laborer']
    ratio_urban = row['ratio_urban']
    
    # 处理未来协变量
    cov_future = cov_scaler.transform(np.array([
        ratio_gdp,
        num_student,
        num_laborer,
        ratio_urban
    ]).reshape(1, -1))

    # 生成预测时间点
    future_progress = np.linspace(0.05, 0.95, 10)
    
    S_f = sigmoid(future_progress, L, x0, k).reshape(-1, 1)
    
    # 复制协变量到每个时间点
    C_f = np.tile(cov_future, (len(future_progress), 1))
    
    # 构建交互特征
    inter_f = []
    for j in range(C_f.shape[1]):
        inter_f.append(S_f * C_f[:, j].reshape(-1, 1))
    inter_f = np.hstack(inter_f) if inter_f else np.empty((len(future_progress), 0))
    
    # 构建预测特征矩阵
    X_f = np.hstack([S_f, C_f, inter_f])

    # 预测就业率
    y_f = model.predict(X_f)

    # 生成预测日期
    start = pd.to_datetime(f"{future_year}-01-01")
    end = pd.to_datetime(f"{future_year}-08-01")
    future_dates = [start + (end - start) * p for p in future_progress]

    # 创建预测结果DataFrame
    year_predictions = pd.DataFrame({
        'year': future_year,
        'date': future_dates, 
        'pred_rate': y_f,
        'progress': future_progress
    })
    
    # 添加到总预测结果
    all_predictions = pd.concat([all_predictions, year_predictions])

# 输出所有预测结果
print("\n2025和2026年预测结果:")
print(all_predictions)

# 保存预测结果
with open('./Q2_preds.md','w') as f:
    # 添加数据来源说明
    f.write("## 就业率预测结果\n\n")
    f.write("**数据来源:**\n")
    for col in predict_data.columns[1:]:
        source = predict_data[col].iloc[-1]  # 获取最后一行作为来源
        f.write(f"- {col}: {source}\n")
    f.write("\n")
    
    # 格式化输出预测结果
    output_df = all_predictions[['year', 'date', 'pred_rate']].copy()
    output_df['date'] = output_df['date'].dt.strftime('%Y-%m-%d')
    output_df['pred_rate'] = output_df['pred_rate'].round(2)
    f.write(output_df.to_markdown(index=False))
    print("\n预测结果已保存到 Q2_preds.md")

# 可视化所有预测结果
plt.figure(figsize=(12, 8))
plt.scatter(df['date'], df['rate'], alpha=0.5, label='历史数据')

# 为每个预测年份绘制不同的颜色
colors = ['#FF5733', '#900C3F']  # 2025: 橙红色, 2026: 深红色
markers = ['o', 's']  # 2025: 圆圈, 2026: 方块

for i, year in enumerate(future_years):
    year_data = all_predictions[all_predictions['year'] == year]
    plt.plot(year_data['date'], year_data['pred_rate'], 
             marker=markers[i], linestyle='-', 
             color=colors[i], 
             markersize=8,
             label=f'{year}年预测')

plt.xlabel('日期')
plt.ylabel('就业率 (%)')
plt.title('2025和2026年就业率预测')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./Q2_prediction.png', dpi=300)
plt.close()
print("预测图表已保存到 Q2_prediction.png")

# 创建预测结果趋势图（按进度）
plt.figure(figsize=(12, 8))

# 绘制历史数据
plt.scatter(df['progress'], df['rate'], alpha=0.7, label='历史数据', color='#3498DB')

# 绘制预测数据
for i, year in enumerate(future_years):
    year_data = all_predictions[all_predictions['year'] == year]
    plt.plot(year_data['progress'], year_data['pred_rate'], 
             color=colors[i], linewidth=2.5,
             label=f'{year}年预测趋势')
    
    # 标记预测点
    plt.scatter(year_data['progress'], year_data['pred_rate'], 
                color=colors[i], s=100, marker=markers[i],
                label=f'{year}年预测点')

plt.xlabel('时间进度 (0-1)')
plt.ylabel('就业率 (%)')
plt.title('2025和2026年就业率预测趋势')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./Q2_trend_prediction.png', dpi=300)
plt.close()
print("趋势预测图表已保存到 Q2_trend_prediction.png")

# 创建预测结果表格图
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# 准备表格数据
table_data = []
for _, row in all_predictions.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    table_data.append([
        str(int(row['year'])),
        date_str,
        f"{row['pred_rate']:.2f}%"
    ])

# 创建表格
table = ax.table(
    cellText=table_data,
    colLabels=['年份', '日期', '预测就业率'],
    cellLoc='center',
    loc='center'
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)  # 调整表格大小

# 添加标题
plt.title('2025和2026年就业率预测结果', fontsize=14, pad=20)

# 保存表格图
plt.savefig('./Q2_prediction_table.png', bbox_inches='tight', dpi=300)
plt.close()
print("预测结果表格已保存到 Q2_prediction_table.png")

print("\n预测完成!")

# ============================
# 疫情分组分析部分
# ============================
print("\n=== 疫情分组分析 ===")

# 按年份独立拟合
years = sorted(df['year'].unique())
params_by_year = {}

plt.figure(figsize=(15, 10))
plt.suptitle('按年份独立拟合就业率趋势', fontsize=16)

for idx, year in enumerate(years):
    year_data = df[df['year'] == year].copy()
    if len(year_data) < 5:
        print(f"跳过{year}年，数据点不足")
        continue
        
    x_data = year_data['progress'].values
    y_data = year_data['rate'].values
    
    # 初始参数估计
    A_est = y_data.min()
    K_est = y_data.max()
    x0_est = np.median(x_data)
    k_est = 5.0
    
    # 设置合理的参数边界
    lower_bounds = [max(0, A_est-20), max(0, K_est-20), 0.1, 0]
    upper_bounds = [min(100, A_est+20), min(100, K_est+20), 15, 1]
    
    try:
        params, _ = curve_fit(
            four_pl,
            x_data,
            y_data,
            p0=[A_est, K_est, k_est, x0_est],
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
        params_by_year[year] = params
        
        # 绘制子图
        plt.subplot(3, 3, idx+1)
        plt.scatter(x_data, y_data, label=f'{year}年数据')
        
        x_fit = np.linspace(0, 1, 100)
        y_fit = four_pl(x_fit, *params)
        plt.plot(x_fit, y_fit, 'r-', label='拟合曲线')
        
        plt.title(f'{year}年就业率拟合')
        plt.xlabel('时间进度')
        plt.ylabel('就业率 (%)')
        plt.legend()
        plt.grid(True)
        
        # 计算拟合质量指标
        residuals = y_data - four_pl(x_data, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print(f"{year}年拟合结果: A={params[0]:.2f}, K={params[1]:.2f}, k={params[2]:.4f}, x0={params[3]:.4f}, R²={r_squared:.3f}")
        
    except Exception as e:
        print(f"无法拟合{year}年: {e}")
        params_by_year[year] = None
        # 只绘制散点图
        plt.subplot(3, 3, idx+1)
        plt.scatter(x_data, y_data, label=f'{year}年数据')
        plt.title(f'{year}年数据分布')
        plt.xlabel('时间进度')
        plt.ylabel('就业率 (%)')
        plt.legend()
        plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./yearly_fitting.png')
plt.close()

# 按疫情分组拟合
group_params = {}
group_ci = {}
groups = ["疫情前 (2016-2019)", "疫情期间 (2020-2022)"]

plt.figure(figsize=(14, 6))

for group_idx, group in enumerate(groups):
    group_data = df[df['group'] == group].copy()
    group_data['progress'] = np.clip(group_data['progress'], 0, 1)
    
    # 排除极端异常值
    q_low = group_data['rate'].quantile(0.05)
    q_high = group_data['rate'].quantile(0.95)
    group_data = group_data[(group_data['rate'] >= q_low) & (group_data['rate'] <= q_high)]
    
    if len(group_data) < 10:
        print(f"{group}组数据点不足，跳过拟合")
        continue
        
    x_data = group_data['progress'].values
    y_data = group_data['rate'].values
    
    # 初始参数估计
    A_est = np.percentile(y_data, 5)
    K_est = np.percentile(y_data, 95)
    x0_est = np.median(x_data)
    k_est = 5.0 if group == "疫情前 (2016-2019)" else 2.0
    
    # 设置合理的参数边界
    lower_bounds = [A_est - 10, K_est - 10, 0.1, 0]
    upper_bounds = [A_est + 10, K_est + 10, 15, 1]
    
    try:
        params, pcov = curve_fit(
            four_pl,
            x_data,
            y_data,
            p0=[A_est, K_est, k_est, x0_est],
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
        group_params[group] = params
        
        # 计算拟合质量指标
        residuals = y_data - four_pl(x_data, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 可视化
        plt.subplot(1, len(groups), group_idx+1)
        plt.scatter(x_data, y_data, alpha=0.6, label=f"{group}数据点")
        
        x_fit = np.linspace(0, 1, 100)
        y_fit = four_pl(x_fit, *params)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f"{group}拟合曲线")
        
        plt.title(f"{group}就业率变化拟合 (R²={r_squared:.3f})")
        plt.xlabel("时间进度 (0-1)")
        plt.ylabel("就业率 (%)")
        plt.legend()
        plt.grid(True)
        
        print(f"{group}拟合结果: A={params[0]:.2f}, K={params[1]:.2f}, k={params[2]:.4f}, x0={params[3]:.4f}, R²={r_squared:.3f}")
        
        # 计算95%置信区间
        n_simulations = 1000
        y_sims = np.zeros((len(x_fit), n_simulations))
        
        for i in range(n_simulations):
            random_params = np.random.multivariate_normal(params, pcov)
            y_sims[:, i] = four_pl(x_fit, *random_params)
        
        ci_lower = np.percentile(y_sims, 2.5, axis=1)
        ci_upper = np.percentile(y_sims, 97.5, axis=1)
        
        plt.fill_between(x_fit, ci_lower, ci_upper, color='r', alpha=0.2, label='95%置信区间')
        
    except Exception as e:
        print(f"无法拟合 {group} 组: {e}")
        group_params[group] = None
        group_ci[group] = None
        # 只绘制散点图
        plt.subplot(1, len(groups), group_idx+1)
        plt.scatter(x_data, y_data, alpha=0.6, label=f"{group}数据点")
        plt.title(f"{group}数据分布")
        plt.xlabel("时间进度 (0-1)")
        plt.ylabel("就业率 (%)")
        plt.legend()
        plt.grid(True)

plt.tight_layout()
plt.savefig('./Q2_covid_analysis.png')
plt.close()

# 分组拟合结果分析
if len(group_params) == 2:
    pre_covid_params = group_params.get("疫情前 (2016-2019)")
    covid_params = group_params.get("疫情期间 (2020-2022)")
    
    if pre_covid_params is not None and covid_params is not None:
        print("\n分组拟合参数对比分析:")
        print(f"{'参数':<8} {'疫情前':<15} {'疫情期间':<15} {'变化率(%)':<10}")
        print("-" * 45)
        
        def calc_change(new, old):
            return ((new - old) / old) * 100 if old != 0 else float('nan')
        
        param_names = ["A (初期)", "K (末期)", "k (陡度)", "x₀ (拐点)"]
        
        for i, name in enumerate(param_names):
            pre_val = pre_covid_params[i]
            covid_val = covid_params[i]
            change = calc_change(covid_val, pre_val)
            print(f"{name:<8} {pre_val:.3f}{'':<10} {covid_val:.3f}{'':<10} {change:.2f}%")
        
        # 特别分析趋势变化
        print("\n疫情对就业趋势的影响分析:")
        print("1. 就业起点变化 (A):", 
              "上升" if covid_val > pre_val else "下降", 
              f"({change:.2f}%)")
        
        print("2. 就业终点变化 (K):", 
              "上升" if covid_val > pre_val else "下降", 
              f"({change:.2f}%)")
        
        print("3. 就业变化陡度 (k):", 
              "变陡" if covid_val > pre_val else "变缓", 
              f"({change:.2f}%)")
        
        print("4. 拐点位置变化 (x₀):", 
              "推后" if covid_val > pre_val else "提前", 
              f"({change:.2f}%)")

        # 计算模型差异
        x_eval = np.linspace(0, 1, 10)
        pre_pred = four_pl(x_eval, *pre_covid_params)
        covid_pred = four_pl(x_eval, *covid_params)
        avg_change = np.mean(covid_pred - pre_pred)
        max_change = np.max(np.abs(covid_pred - pre_pred))
        
        print(f"\n综合影响: 平均就业率变化: {avg_change:.2f}%")
        print(f"最大变化幅度: {max_change:.2f}%")
else:
    print("\n警告: 未获得完整分组拟合结果，跳过分析")
    

# ============================
# 各年份拟合曲线对比图
# ============================
plt.figure(figsize=(12, 8))

# 创建颜色映射 (2016-2022)
years = sorted(df['year'].unique())
color_map = plt.get_cmap('viridis')
year_colors = {year: color_map(i/(len(years)-1)) for i, year in enumerate(years)}

# 创建图例元素
legend_elements = []

# 绘制所有年份拟合曲线
for year in years:
    if year in params_by_year and params_by_year[year] is not None:
        # 获取拟合参数
        A, K, k, x0 = params_by_year[year]
        
        # 生成拟合曲线
        x_range = np.linspace(0, 1, 100)
        y_fit = four_pl(x_range, A, K, k, x0)
        
        # 绘制曲线
        plt.plot(
            x_range, 
            y_fit, 
            color=year_colors[year], 
            linewidth=3 if year > 2019 else 2,
            linestyle='-' if year <= 2019 else '--'
        )
        
        # 标记特殊年份
        if year == 2020:
            plt.text(
                x_range[10], 
                y_fit[10], 
                '2020(疫情开始)', 
                fontsize=10,
                color=year_colors[year],
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        # 添加图例元素
        legend_elements.append(Line2D(
            [0], [0], 
            color=year_colors[year], 
            lw=2, 
            label=f'{year}年拟合曲线'
        ))

# 添加实际数据点
for year in years:
    year_data = df[df['year'] == year]
    plt.scatter(
        year_data['progress'],
        year_data['rate'],
        color=year_colors[year],
        alpha=0.8,
        s=60,
        marker='o' if year <= 2019 else 's'
    )

# 添加疫情时间线
plt.axvline(x=0.8, color='gray', linestyle=':', linewidth=2)
plt.text(0.81, plt.ylim()[1]*0.95, '疫情开始(2020)', fontsize=12, color='gray')

# 设置图表属性
plt.title('2016-2022年就业率趋势变化对比', fontsize=16)
plt.xlabel('时间进度 (0-1)')
plt.ylabel('就业率 (%)')
plt.grid(True, linestyle='--', alpha=0.7)

# 创建复合图例
legend_elements += [
    Line2D([0], [0], marker='o', color='w', label='2016-2019年实际数据点',
           markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='2020-2022年实际数据点',
           markerfacecolor='gray', markersize=10),
    Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='疫情时间线')
]

plt.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

# 突出关键区域
plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]], x1=0.7, x2=0.9, 
                  color='lightpink', alpha=0.2)
plt.text(0.72, plt.ylim()[0]+1, '关键转折区域', fontsize=10, color='darkred')

# 保存对比图
plt.tight_layout()
plt.savefig('./Q2_yearly_comparison.png', dpi=300)
plt.close()

# ============================
# 疫情前后拟合曲线对比（同图叠加）
# ============================
plt.figure(figsize=(12, 8))

# 专业颜色设计
group_colors = {
    "疫情前 (2016-2019)": "#3498db",  # 专业蓝色
    "疫情期间 (2020-2022)": "#e74c3c"  # 专业红色
}

# 置信区间透明度设置
ci_alpha = 0.2

# 首先绘制置信区间（底层）
for group in groups:
    if group in group_ci and group_ci[group] is not None and group in group_params:
        # 获取拟合结果
        params = group_params[group]
        x_fit = np.linspace(0, 1, 100)
        ci_lower, ci_upper = group_ci[group]
        
        # 绘制置信区间（底层）
        plt.fill_between(
            x_fit,
            ci_lower,
            ci_upper,
            color=group_colors[group],
            alpha=ci_alpha,
            label=f'{group}95%置信区间'
        )

# 然后绘制拟合曲线和数据点
for group in groups:
    if group in group_params and group_params[group] is not None:
        group_data = df[df['group'] == group]
        params = group_params[group]
        
        # 绘制数据点（带大小和颜色）
        plt.scatter(
            group_data['progress'],
            group_data['rate'],
            color=group_colors[group],
            alpha=0.7,
            s=70,
            edgecolor='white',
            linewidth=1.5,
            label=f'{group}数据点',
            zorder=5
        )
        
        # 绘制拟合曲线（加粗）
        x_fit = np.linspace(0, 1, 100)
        y_fit = four_pl(x_fit, *params)
        plt.plot(
            x_fit,
            y_fit,
            color=group_colors[group],
            linewidth=3,
            linestyle='-',
            label=f'{group}拟合曲线',
            zorder=4
        )

# 专业样式设置
plt.title('疫情前后就业率变化趋势对比', fontsize=18, pad=15)
plt.xlabel('时间进度 (0-1)', fontsize=14)
plt.ylabel('就业率 (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)

# 图例处理
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(), 
    by_label.keys(), 
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    fancybox=True,
    shadow=True,
    ncol=3,
    fontsize=12
)

# 轴范围设置
plt.xlim(-0.05, 1.05)
plt.ylim(max(0, df['rate'].min()-10), min(100, df['rate'].max()+10))

# 添加网格和参考线
plt.axhline(y=50, color='gray', linestyle='-', alpha=0.2)
plt.axvline(x=0.5, color='gray', linestyle='-', alpha=0.2)

# 保存图片
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部图例留空间
plt.savefig('./Q2_covid_comparison_professional.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n所有分析完成!")