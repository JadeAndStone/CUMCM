import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

    plt.savefig("./Graphs/" + subject + "采样年际变化.png")
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
    plt.savefig("./Graphs/" + subject + "参数年际变化.png")


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
            alpha=0.6,
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
                "--",
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


def function(subject):
    data = dataimport(subject)  # 可以替换为"硕士"或"博士"
    ddd, params_by_year = fit_by_year(data)
    print(ddd)
    draw_attributes(ddd, params_by_year, subject)
    group_params, group_ci = fit_by_group(data)
    show_comparison(data, params_by_year, group_params, group_ci, subject)
    review(data, group_params)

    plt.show()


def main():
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # function("本科")
    # function("硕士")
    # function("博士")
    function("整体")


if __name__ == "__main__":
    main()
