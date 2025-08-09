import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
import datetime


years = range(2016, 2023)
groups = ["Pre-COVID (2016-2019)", "COVID (2020-2022)"]


def dataimport(subject):
    multisheet = pd.read_excel("1_university_employment_cleaned.xlsx", sheet_name=None)
    data = pd.DataFrame()
    for year, df in multisheet.items():
        data = pd.concat([data, df], ignore_index=True)

    data["day_of_year"] = data["日期"].dt.dayofyear
    data["employment_rate"] = data[subject]
    data["year"] = data["日期"].dt.year

    data = data[["year", "day_of_year", "employment_rate"]]  # 只有年，天数和整体
    data = data[~((data["year"] == 2020) & (data["day_of_year"] < 60))]
    # 创建分组标签
    data["group"] = data["year"].apply(
        lambda y: "Pre-COVID (2016-2019)" if y <= 2019 else "COVID (2020-2022)"
    )
    return data


def four_pl(x, A, K, k, x0):
    """四参数逻辑函数"""
    return A + (K - A) / (1 + np.exp(-k * (x - x0)))


def fit_by_year(data):
    # ======================
    # 方法1: 按年份独立拟合
    # ======================
    print("按年份独立拟合结果:")
    params_by_year = {}

    for year in years:
        year_data = data[data["year"] == year]
        x_data = year_data["day_of_year"].values
        y_data = year_data["employment_rate"].values

        # 初始参数估计
        early_data = y_data[x_data <= np.quantile(x_data, 0.25)]
        A_est = np.mean(early_data) if len(early_data) > 0 else np.min(y_data)

        late_data = y_data[x_data >= np.quantile(x_data, 0.75)]
        K_est = np.mean(late_data) if len(late_data) > 0 else np.max(y_data)

        x0_est = np.median(x_data)
        k_est = 0.05

        bounds = ([0, 0, 0.01, min(x_data)], [1, 1, 0.5, max(x_data)])
        print(A_est, K_est, k_est, x0_est)
        try:
            params, _ = curve_fit(
                four_pl,
                x_data,
                y_data,
                p0=[A_est, K_est, k_est, x0_est],
                bounds=bounds,
                maxfev=10000,
            )
            params_by_year[year] = params
            print(
                f"{year}: A={params[0]:.2f}, K={params[1]:.2f}, k={params[2]:.4f}, x0={params[3]:.1f}"
            )
        except Exception as e:
            print(f"Error fitting {year}: {e}")
            params_by_year[year] = None

    ddd = pd.DataFrame(params_by_year).T
    ddd.columns = ["A", "K", "k", "x0"]
    print(ddd)

    return ddd, params_by_year


def draw_attributes(ddd, params_by_year, subject):

    fig, axs = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    # A系列（对数坐标）
    axs[0].plot(ddd.index, ddd["A"], "o-", color="blue")
    axs[0].set_title("A Value")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)

    # K系列
    axs[1].plot(ddd.index, ddd["K"], "s-", color="green")
    axs[1].set_title("K Value")
    axs[1].grid(True, ls="--", alpha=0.5)

    # k系列
    axs[2].plot(ddd.index, ddd["k"], "D-", color="purple")
    axs[2].set_title("k Value")
    axs[2].grid(True, ls="--", alpha=0.5)

    # x0系列
    axs[3].plot(ddd.index, ddd["x0"], "^-", color="red")
    axs[3].set_title("x0 Value")
    axs[3].grid(True, ls="--", alpha=0.5)
    axs[3].set_xlabel("Year")

    sample_days = [
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
    ]  # 你可以根据需要调整采样天数

    plt.savefig("./Graphs/" + subject + "参数年际变化.png")
    plt.figure(figsize=(10, 6))

    for target_day in sample_days:
        value_on_target_day = []
        for year in years:
            params = params_by_year[year]
            if params is not None:
                val = four_pl(target_day, *params)
                value_on_target_day.append(val)
            else:
                value_on_target_day.append(np.nan)
        plt.plot(years, value_on_target_day, marker="o", label=f"{target_day}日")

    plt.xlabel("年份")
    plt.ylabel("就业率拟合值")
    plt.title("不同采样天数各年就业率拟合值趋势")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Graphs/" + subject + "采样年际变化.png")


def fit_by_group(data):
    # ============================
    # 方法2: 按疫情分组拟合
    # ============================
    print("\n按疫情分组拟合结果:")
    group_params = {}
    group_ci = {}

    for group in groups:
        group_data = data[data["group"] == group]
        x_data = group_data["day_of_year"].values
        y_data = group_data["employment_rate"].values

        # 初始参数估计（使用组内所有数据）
        early_data = y_data[x_data <= np.quantile(x_data, 0.25)]
        A_est = np.mean(early_data) if len(early_data) > 0 else np.min(y_data)

        late_data = y_data[x_data >= np.quantile(x_data, 0.75)]
        K_est = np.mean(late_data) if len(late_data) > 0 else np.max(y_data)

        x0_est = np.median(x_data)
        k_est = 0.05 if group == "Pre-COVID (2016-2019)" else 0.03

        bounds = ([0, 0, 0.01, min(x_data)], [1, 1, 0.5, max(x_data)])

        try:
            params, pcov = curve_fit(
                four_pl,
                x_data,
                y_data,
                p0=[A_est, K_est, k_est, x0_est],
                bounds=bounds,
                maxfev=10000,
            )

            group_params[group] = params

            perr = np.sqrt(np.diag(pcov))
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = four_pl(x_fit, *params)

            # 蒙特卡洛模拟参数不确定性
            n_simulations = 1000
            y_sims = np.zeros((len(x_fit), n_simulations))

            for i in range(n_simulations):
                # 从参数分布中随机采样
                random_params = np.random.multivariate_normal(params, pcov)
                y_sims[:, i] = four_pl(x_fit, *random_params)

            # 计算95%置信区间
            ci_lower = np.percentile(y_sims, 2.5, axis=1)
            ci_upper = np.percentile(y_sims, 97.5, axis=1)

            group_ci[group] = (ci_lower, ci_upper)

            print(
                f"{group}: A={params[0]:.2f}, K={params[1]:.2f}, k={params[2]:.4f}, x0={params[3]:.1f}"
            )
        except Exception as e:
            print(f"Error fitting {group}: {e}")
            group_params[group] = None

    return group_params, group_ci


def show_comparison(data, params_by_year, group_params, group_ci, subject):
    # ============================
    # 可视化比较两种拟合方法
    # ============================
    plt.figure(figsize=(10, 11))

    # 1. 按年份独立拟合可视化

    # 定义从冷色到暖色的渐变颜色映射
    # 2016(冷色:蓝绿色) -> 2022(暖色:橙红色)
    cool_color = np.array([0, 0, 255]) / 255  # 蓝绿色
    warm_color = np.array([255, 0, 0]) / 255  # 橙红色
    # 生成7个过渡颜色
    colors = [cool_color + (warm_color - cool_color) * i / 6 for i in range(7)]

    plt.subplot(2, 1, 1)
    for year in years:
        year_data = data[data["year"] == year]
        plt.scatter(
            year_data["day_of_year"],
            year_data["employment_rate"],
            label=f"{year} Data",
            alpha=0.5,
            color=colors[year - 2016],
        )

        if params_by_year[year] is not None:
            A, K, k, x0 = params_by_year[year]
            x_fit = np.linspace(
                min(year_data["day_of_year"]), max(year_data["day_of_year"]), 100
            )
            y_fit = four_pl(x_fit, A, K, k, x0)
            plt.plot(
                x_fit,
                y_fit,
                "-",
                linewidth=2,
                label=f"{year} Fit",
                color=colors[year - 2016],
            )

    plt.title("按年份独立拟合")
    plt.xlabel("Day of Year")
    plt.ylabel("Employment Rate (%)")
    plt.legend()
    plt.grid(True)

    # 2. 按疫情分组拟合可视化
    plt.subplot(2, 1, 2)
    colors = {"Pre-COVID (2016-2019)": "blue", "COVID (2020-2022)": "red"}

    for group in groups:
        group_data = data[data["group"] == group]
        plt.scatter(
            group_data["day_of_year"],
            group_data["employment_rate"],
            color=colors[group],
            alpha=0.6,
            label=f"{group} Data",
        )

        if group in group_params and group_params[group] is not None:
            A, K, k, x0 = group_params[group]
            x_fit = np.linspace(
                min(group_data["day_of_year"]), max(group_data["day_of_year"]), 100
            )
            y_fit = four_pl(x_fit, A, K, k, x0)
            plt.plot(
                x_fit,
                y_fit,
                "-",
                linewidth=3,
                color=colors[group],
                label=f"{group} Fit",
            )
            plt.fill_between(
                x_fit,
                group_ci[group][0],
                group_ci[group][1],
                color=colors[group],
                alpha=0.2,
                label="95%置信区间",
            )

    plt.title("按疫情分组拟合")
    plt.xlabel("Day of Year")
    plt.ylabel("Employment Rate (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./Graphs/" + subject + "两种拟合对比.png")


def review(data, group_params):
    # ============================
    # 分组拟合结果分析
    # ============================
    pre_covid_params = group_params["Pre-COVID (2016-2019)"]
    covid_params = group_params["COVID (2020-2022)"]

    if pre_covid_params is not None and covid_params is not None:
        print("\n分组拟合参数对比:")
        print(f"{'参数':<10} {'Pre-COVID':<15} {'COVID':<15} {'变化 (%)':<10}")
        print("-" * 45)

        A_pre, K_pre, k_pre, x0_pre = pre_covid_params
        A_cov, K_cov, k_cov, x0_cov = covid_params

        # 计算变化百分比
        def calc_change(new, old):
            return (new - old) / old * 100

        print(
            f"{'A':<10} {A_pre:.2f}{'':<8} {A_cov:.2f}{'':<8} {calc_change(A_cov, A_pre):.1f}%"
        )
        print(
            f"{'K':<10} {K_pre:.2f}{'':<8} {K_cov:.2f}{'':<8} {calc_change(K_cov, K_pre):.1f}%"
        )
        print(
            f"{'k':<10} {k_pre:.4f}{'':<5} {k_cov:.4f}{'':<5} {calc_change(k_cov, k_pre):.1f}%"
        )
        print(
            f"{'x0':<10} {x0_pre:.1f}{'':<8} {x0_cov:.1f}{'':<8} {calc_change(x0_cov, x0_pre):.1f}%"
        )


# 拟合变量未来趋势，使用ARIMR
def fit_attribute_ARIMR(ddd, target_attribute, numofyear):
    """使用非季节性ARIMA模型预测未来5年数据

    参数:
        ddd - 包含历史数据的DataFrame
        target_attribute - 要预测的参数名称

    返回:
        包含预测年份和值的DataFrame
    """
    from pmdarima import auto_arima
    import pandas as pd

    # 准备时间序列数据（按年聚合）
    ts_data = ddd[[target_attribute]].dropna().copy()
    ts_data.index = pd.to_datetime(ts_data.index.astype(str) + "-12-31")
    ts_data = ts_data.resample("YE").mean()

    # 自动选择最佳ARIMA参数（非季节性）
    if target_attribute == "A":
        model = auto_arima(
            ts_data,
            seasonal=False,
            stepwise=False,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
            d=1,
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,
            max_order=6,
            with_intercept=False,
            information_criterion="bic",
        )
    elif target_attribute == "K":
        model = auto_arima(
            ts_data,
            seasonal=False,
            stepwise=False,
            start_p=1,
            start_q=1,
            min_p=1,
            min_q=1,
            max_p=5,
            max_q=5,
            d=2,
            test="adf",
            information_criterion="bic",
            trend="t",
            max_order=12,
            with_intercept=False,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
        )
    elif target_attribute == "k":
        model = auto_arima(
            ts_data,
            seasonal=False,
            stepwise=False,
            start_p=1,
            start_q=1,
            min_p=1,
            min_q=1,
            max_p=7,
            max_q=7,
            d=1,
            test="adf",
            information_criterion="bic",
            trend="ct",
            max_order=14,
            with_intercept=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
        )
    elif target_attribute == "x0":
        model = auto_arima(
            ts_data,
            seasonal=False,  # 启用季节性成分
            m=4,  # 设置4年周期
            stepwise=True,
            start_p=1,
            start_q=1,
            max_p=5,  # 降低最大p值
            max_q=5,  # 降低最大q值
            d=1,
            test="adf",
            information_criterion="aicc",  # 使用AICc准则
            trend="c",  # 仅保留常数项
            max_order=10,  # 降低最大阶数
            with_intercept=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
        )

    # 训练模型
    model.fit(ts_data)

    # 预测未来5年
    forecast = model.predict(n_periods=numofyear)

    # 生成预测年份
    last_year = ts_data.index.max().year
    forecast_years = [last_year + i for i in range(1, 6)]

    # 限制K参数预测值不超过1
    # 新增的截断逻辑
    if target_attribute == "K":
        forecast = np.clip(forecast, 0, 1)

    return pd.DataFrame(
        {"year": forecast_years, "value": forecast, "attribute": target_attribute}
    )


def predict_attr(ddd):
    df1 = fit_attribute_ARIMR(ddd, "A", 5)
    df2 = fit_attribute_ARIMR(ddd, "K", 5)
    df3 = fit_attribute_ARIMR(ddd, "k", 5)
    df4 = fit_attribute_ARIMR(ddd, "x0", 5)

    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Attribute Values by Year", fontsize=16)

    # 绘制第一个子图 (A)
    axes[0, 0].plot(df1["year"], df1["value"], marker="o", color="blue")
    axes[0, 0].set_title("Attribute A")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].grid(True, linestyle="--", alpha=0.7)

    # 绘制第二个子图 (K)
    axes[0, 1].plot(df2["year"], df2["value"], marker="s", color="green")
    axes[0, 1].set_title("Attribute K")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].grid(True, linestyle="--", alpha=0.7)
    axes[0, 1].set_ylim(0.95, 1.05)  # 调整Y轴范围以突出变化

    # 绘制第三个子图 (k)
    axes[1, 0].plot(df3["year"], df3["value"], marker="^", color="red")
    axes[1, 0].set_title("Attribute k")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].grid(True, linestyle="--", alpha=0.7)

    # 绘制第四个子图 (x0)
    axes[1, 1].plot(df4["year"], df4["value"], marker="d", color="purple")
    axes[1, 1].set_title("Attribute x0")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True, linestyle="--", alpha=0.7)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间

    dfA = fit_attribute_ARIMR(ddd, "A", 5)
    dfK = fit_attribute_ARIMR(ddd, "K", 5)
    dfk = fit_attribute_ARIMR(ddd, "k", 5)
    dfx0 = fit_attribute_ARIMR(ddd, "x0", 5)

    # 将预测结果转换为宽格式
    pred_A = dfA.set_index("year")["value"].rename("A")
    pred_K = dfK.set_index("year")["value"].rename("K")
    pred_k = dfk.set_index("year")["value"].rename("k")
    pred_x0 = dfx0.set_index("year")["value"].rename("x0")

    # 合并预测数据
    pred_df = pd.concat([pred_A, pred_K, pred_k, pred_x0], axis=1)

    # 与历史数据拼接
    fff = pd.concat([ddd, pred_df])

    # 按年份排序并填充可能的空值
    fff = fff.sort_index().fillna(method="ffill")

    plt.figure(figsize=(15, 10))

    attributes = ["A", "K", "k", "x0"]
    for i, attr in enumerate(attributes, 1):
        plt.subplot(2, 2, i)

        # 绘制历史数据（2016-2022）
        plt.plot(
            fff.loc[:2022].index,
            fff.loc[:2022][attr],
            "o-",
            label="历史数据",
            markersize=5,
        )

        # 绘制预测数据（2023-2027）
        plt.plot(
            fff.loc[2023:].index,
            fff.loc[2023:][attr],
            "s--",
            color="red",
            label="预测数据",
            markersize=5,
        )

        plt.title(f"{attr}参数变化趋势")
        plt.xlabel("年份")
        plt.ylabel("参数值")
        plt.xticks(range(2016, 2028, 2))
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    print(pred_A)
    print(pred_K)
    print(pred_k)
    print(pred_x0)

    # 生成预测曲线
    plt.figure(figsize=(12, 8))
    x = np.linspace(0, 365, 100)

    for year in range(2023, 2028):
        params = fff.loc[year][["A", "K", "k", "x0"]]
        y = four_pl(x, *params)
        plt.plot(x, y, label=f"{year}年预测", linestyle="--" if year > 2022 else "-")

    # 绘制历史基准曲线（2022年）
    hist_params = fff.loc[2022][["A", "K", "k", "x0"]]
    y_hist = four_pl(x, *hist_params)
    plt.plot(x, y_hist, "k-", linewidth=2, label="2022基准")

    plt.title("就业率预测曲线（2023-2027）")
    plt.xlabel("年积日")
    plt.ylabel("就业率(%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fff


def show_prediction_comparison(params_by_year, pred_data, subject, data):
    plt.figure(figsize=(12, 8))
    x = np.linspace(0, 365, 100)
    # 生成渐变色（浅绿→深绿）
    pred_colors = plt.cm.GnBu(np.linspace(0.9, 0.5, len(pred_data)))

    # 定义从冷色到暖色的渐变颜色映射
    # 2016(冷色:蓝绿色) -> 2022(暖色:橙红色)
    cool_color = np.array([0, 0, 255]) / 255  # 蓝绿色
    warm_color = np.array([255, 0, 0]) / 255  # 橙红色
    # 生成7个过渡颜色
    colors = [cool_color + (warm_color - cool_color) * i / 6 for i in range(7)]

    for year in years:
        year_data = data[data["year"] == year]
        plt.scatter(
            year_data["day_of_year"],
            year_data["employment_rate"],
            label=f"{year} Data",
            alpha=0.2,
            color=colors[year - 2016],
        )

        if params_by_year[year] is not None:
            A, K, k, x0 = params_by_year[year]
            x_fit = np.linspace(
                min(year_data["day_of_year"]), max(year_data["day_of_year"]), 100
            )
            y_fit = four_pl(x_fit, A, K, k, x0)
            plt.plot(
                x_fit,
                y_fit,
                "-",
                linewidth=1,
                label=f"{year} Fit",
                color=colors[year - 2016],
                alpha=0.7,
            )
    # 绘制预测曲线（2023-2027）

    x_pred = np.linspace(min(data["day_of_year"]), max(data["day_of_year"]), 100)
    for i, (year, row) in enumerate(pred_data.iterrows()):
        y_pred = four_pl(x_pred, row["A"], row["K"], row["k"], row["x0"])
        plt.plot(
            x_pred,
            y_pred,
            color=pred_colors[i],
            linewidth=2.5,
            linestyle="--",
            label=f"{year}预测",
        )

    # 统一可视化样式
    plt.title(f"{subject}就业率历史拟合与预测对比", fontsize=14)
    plt.xlabel("年积日", fontsize=12)
    plt.ylabel("就业率(%)", fontsize=12)
    plt.legend(ncol=2, loc="upper left", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./Graphs/{subject}_full_comparison.png", dpi=300, bbox_inches="tight")


def function(subject):
    data = dataimport(subject)  # 可以替换为"硕士"或"博士"
    ddd, params_by_year = fit_by_year(data)
    print(ddd)
    draw_attributes(ddd, params_by_year, subject)
    group_params, group_ci = fit_by_group(data)
    show_comparison(data, params_by_year, group_params, group_ci, subject)
    review(data, group_params)
    fff = predict_attr(ddd)
    show_prediction_comparison(params_by_year, fff.loc[2023:2027], subject, data)

    predict_xtest = range(80, 180, 10)

    predict_df = pd.DataFrame(columns=["时间", "就业率"])

    for year in [2025, 2026]:
        params = fff.loc[year][["A", "K", "k", "x0"]]
        for x in predict_xtest:
            # 计算就业率
            rate = round(four_pl(x, *params) * 100, 1)

            # 生成日期（示例：20250101）
            date_obj = datetime.datetime(year, 1, 1) + datetime.timedelta(days=x - 1)
            date_str = date_obj.strftime("%Y%m%d")

            # 添加到数据框
            predict_df.loc[date_str] = [date_str, rate]

    # 设置索引并重命名列
    predict_df.index.name = "日期"
    predict_df.columns = [
        "时间（格式为20230101，即年份+月份+日期）",
        "就业率（%），保留一位小数",
    ]

    # 保存结果
    predict_df.to_excel(f"./附件二：单变量预测.xlsx", index=False)
    print(predict_df.head())

    plt.show()

    # 创建日期索引（示例：假设年份为2023，实际会根据预测年份变化）


def main():
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # function("本科")
    # function("硕士")
    # function("博士")
    function("整体")


if __name__ == "__main__":
    main()
