import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error


#设置全局宇体为中文
plt.rcParams ['font.sans-serif'] = ['SimHei']
#使用 SimHei 字体
plt.rcParams ['axes.unicode_minus'] = False
#正确显示负号

# 1. 数据整合与预处理
def prepare_data(file_path):
    """读取清洗后的数据并整合为统一时间序列"""
    xls = pd.ExcelFile(file_path)
    all_data = pd.DataFrame()
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df['日期'] = pd.to_datetime(df['日期'])
        all_data = pd.concat([all_data, df[['日期', '整体']]], axis=0)
    
    # 按日期排序并设置索引
    all_data.sort_values('日期', inplace=True)
    all_data.set_index('日期', inplace=True)
    
    # 重采样为日频数据并插值
    daily_data = all_data.resample('D').mean()
    daily_data['整体'] = daily_data['整体'].interpolate(method='time')
    
    return daily_data

# 2. 时间序列平稳性检验
def check_stationarity(series):
    """使用ADF检验时间序列平稳性"""
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    return result[1] < 0.05  # 返回是否平稳

# 3. 自动ARIMA模型选择
def select_arima_model(data):
    """使用auto_arima自动选择最佳ARIMA参数"""
    model = auto_arima(
        data,
        seasonal=True,
        m=12,  # 年度季节性周期
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )
    print(model.summary())
    return model

# 4. 模型训练与预测
def train_and_predict(data, model_order, forecast_periods=730):
    """训练SARIMA模型并预测未来两年"""
    # 划分训练集（2016-2022）
    train = data.loc[:'2022-12-31']
    
    # 模型训练
    model = SARIMAX(
        train,
        order=model_order[:3],
        seasonal_order=model_order[3:],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # 未来两年预测（2025-2026）
    end_date = pd.to_datetime('2026-12-31')
    forecast_steps = (end_date - train.index[-1]).days
    forecast = results.get_forecast(steps=forecast_steps)
    
    # 提取预测结果（每月第一个数据点）
    forecast_df = forecast.conf_int()
    forecast_df['预测值'] = forecast.predicted_mean
    
    # 筛选2025-2026年的每月第一天
    forecast_2025_2026 = forecast_df.loc['2025-01-01':'2026-12-01'].resample('MS').first()
    
    return results, forecast_2025_2026

# 主流程
def main():
    # 数据准备
    file_path = '1_university_employment_cleaned.xlsx'
    data = prepare_data(file_path)
    
    # 平稳性检验
    print("平稳性检验结果:")
    is_stationary = check_stationarity(data['整体'])
    print(f"时间序列平稳: {is_stationary}")
    
    # 自动选择ARIMA模型
    print("\n自动ARIMA模型选择:")
    model = select_arima_model(data['整体'])
    model_order = model.order
    seasonal_order = model.seasonal_order
    
    # 模型训练与预测
    results, forecast_results = train_and_predict(
        data['整体'], 
        model_order + seasonal_order
    )
    
    # 可视化结果
    plt.figure(figsize=(14, 7))
    plt.plot(data['整体'], label='历史数据', alpha=0.7)
    plt.plot(results.fittedvalues, label='模型拟合', color='red', alpha=0.7)
    plt.plot(forecast_results['预测值'], label='2025-2026预测', color='green', marker='o')
    plt.fill_between(
        forecast_results.index,
        forecast_results['lower 整体'],
        forecast_results['upper 整体'],
        color='green',
        alpha=0.1
    )
    plt.title('高校毕业生就业率时间序列分析与预测')
    plt.xlabel('日期')
    plt.ylabel('就业率')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 保存预测结果到附件二
    output = forecast_results[['预测值']].reset_index()
    output.columns = ['日期', '预测就业率']
    output.to_excel('附件二_就业率预测结果.xlsx', index=False)
    
    print("\n预测结果已保存到附件二")
    print(output.head(12))  # 显示2025年前半年预测

if __name__ == "__main__":
    main()