import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime

#设置全局宇体为中文
plt.rcParams ['font.sans-serif'] = ['SimHei']
#使用 SimHei 字体
plt.rcParams ['axes.unicode_minus'] = False
#正确显示负号

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
        spring_festival = spring_festival_dates[year - 1]
    else:
        spring_festival = spring_festival_dates[year]
    
    return (date - spring_festival).days

# 读取清洗后的数据
def load_cleaned_data(file_path):
    # 读取所有年份的数据表
    all_data = pd.read_excel(file_path, sheet_name=None)
    
    # 合并所有年份的数据
    combined_df = pd.DataFrame()
    for year, df in all_data.items():
        # 确保年份是整数类型
        df['年份'] = df['年份'].astype(int)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # 添加一年中的第几天特征
    #combined_df['day_of_year'] = combined_df['日期'].dt.dayofyear
    combined_df['day_of_year'] = combined_df['日期'].apply(get_lunar_day_of_year)
    print(combined_df['day_of_year'])
    
    # 排序
    combined_df = combined_df.sort_values(['年份', 'day_of_year'])
    return combined_df

# 高斯过程回归建模函数
def gpr_modeling(df, target_column):
    # 选择特征和目标
    X = df[['年份', 'day_of_year']].values
    y = df[target_column].values
    
    # 标准化处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练测试集 (使用时间序列分割更合适，这里简化处理)
    # 保留最后一年作为测试集
    train_df = df[df['年份'] < 2022]
    test_df = df[df['年份'] == 2022]
    
    if len(train_df) == 0 or len(test_df) == 0:
        # 如果没有2022年数据，使用随机分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # 使用基于时间的分割
        X_train = train_df[['年份', 'day_of_year']].values
        y_train = train_df[target_column].values
        X_test = test_df[['年份', 'day_of_year']].values
        y_test = test_df[target_column].values
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    # 定义高斯过程回归的核心核函数
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 30.0]) + WhiteKernel(noise_level=0.1)
    
    # 创建并训练GPR模型
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=0.01
    )
    
    # 训练模型
    gpr.fit(X_train, y_train)
    
    # 预测
    predict_result = gpr.predict(X_test, return_std=True)
    if isinstance(predict_result, tuple) and len(predict_result) == 2:
        y_pred, y_std = predict_result
    else:
        y_pred = predict_result
        y_std = np.zeros_like(y_pred)  # 如果没有标准差，创建零数组
    
    # 评估模型
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model for {target_column} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # 以下是置信概率为99%的可视化代码
    # # 可视化结果
    # plt.figure(figsize=(12, 6))
    # plt.scatter(range(len(y_test)), y_test, c='k', label='Actual')
    # plt.plot(range(len(y_pred)), y_pred, 'b-', label='Predicted') 
    # plt.fill_between(
    #     range(len(y_pred)),
    #     y_pred - 2.576 * y_std,  # 修改关键参数
    #     y_pred + 2.576 * y_std,  # 修改关键参数
    #     alpha=0.2,
    #     color='blue',
    #     label='99% confidence interval'  # 更新标签
    # )
    # plt.title(f"GPR Prediction for {target_column}")
    # plt.xlabel("Sample index")
    # plt.ylabel(target_column)
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # 以下是置信概率为95%的可视化代码
    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(y_test)), y_test, c='k', label='Actual')
    plt.plot(range(len(y_pred)), y_pred, 'b-', label='Predicted')
    plt.fill_between(
        range(len(y_pred)),
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        alpha=0.2,
        color='blue',
        label='95% confidence interval'
    )
    plt.title(f"GPR Prediction for {target_column}")
    plt.xlabel("Sample index")
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return gpr, scaler, rmse, r2

# 主程序
def main():
    # 加载清洗后的数据
    data = load_cleaned_data('1_university_employment_cleaned.xlsx')
    
    # 为每个目标变量创建GPR模型
    targets = ['本科', '硕士', '博士', '整体']
    models = {}
    
    for target in targets:
        print(f"\n{'='*40}")
        print(f"Training GPR model for {target}")
        print(f"{'='*40}")
        
        # 筛选出包含目标变量的非缺失值
        model_data = data.dropna(subset=[target])
        
        if len(model_data) < 10:
            print(f"Not enough data for {target} modeling. Skipping...")
            continue
            
        # 训练模型
        model, scaler, rmse, r2 = gpr_modeling(model_data, target)
        
        # 保存模型
        models[target] = {
            'model': model,
            'scaler': scaler,
            'rmse': rmse,
            'r2': r2
        }
    
    # 保存所有模型
    joblib.dump(models, 'gpr_employment_models.pkl')
    print("All models saved to gpr_employment_models.pkl")
    
    # 使用模型进行预测
    predict_new_data(models)

# 预测新数据函数
def predict_new_data(models):
    # 创建新的预测数据
    years = [2023]  # 预测2023年
    day_range = range(0, 200, 10)  # 选择要预测的时间点
    
    # 存储所有预测结果
    predictions = {}
    
    for target, model_info in models.items():
        model = model_info['model']
        scaler = model_info['scaler']
        
        target_preds = []
        
        for year in years:
            # 创建预测数据点
            X_pred = np.array([[year, doy] for doy in day_range])
            X_pred_scaled = scaler.transform(X_pred)
            
            # 预测
            y_pred, y_std = model.predict(X_pred_scaled, return_std=True)
            
            # 存储预测结果
            for idx, doy in enumerate(day_range):
                target_preds.append({
                    '年份': year,
                    'day_of_year': doy,
                    '预测值': y_pred[idx],
                    '标准差': y_std[idx]
                })
        
        predictions[target] = pd.DataFrame(target_preds)
    


    # 将预测结果导出为Excel
    with pd.ExcelWriter('employment_predictions_2023.xlsx') as writer:
        for target, df in predictions.items():
            df.to_excel(writer, sheet_name=f"{target}_预测", index=False)
    
    print("预测结果已保存到 employment_predictions_2023.xlsx")

if __name__ == "__main__":
    main()