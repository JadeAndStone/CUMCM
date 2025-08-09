import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from datetime import datetime


def ratio_compute(df, year):
    df.fillna(0, inplace=True)  # 填充缺失值为0
    # print(df)
    print(year)
    # 通过各个就业率和整体就业率倒推人数比例
    X = df[["本科", "硕士", "博士", "高职"]].values  # 转换为numpy数组
    y = df["整体"].values

    # 定义目标函数（残差平方和）
    def objective(w, X, y):
        predictions = np.dot(X, w)
        return np.sum((y - predictions) ** 2)

    # 约束：权重和等于1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    if year <= 2018:
        # 初始权重猜测（均匀分布，简单起见）
        initial_w = [0.25, 0.25, 0.25, 0.25]  # [0.25, 0.25, 0.25, 0.25]
        # 边界：权重非负
        bounds = [(0, 1) for _ in range(4)]  # 每个权重在[0,1]范围内

    else:
        # 初始权重猜测（均匀分布，简单起见）
        initial_w = [0.25, 0.25, 0.25, 0]  # [0.25, 0.25, 0.25, 0.25]
        # 边界：权重非负
        bounds = [
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 0),
        ]  # 每个权重在[0,1]范围内

    # 求解优化问题
    result = minimize(
        objective,
        initial_w,
        args=(X, y),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    # 提取估计的权重
    estimated_weights = result.x
    w_b, w_m, w_d, w_v = estimated_weights  # 对应本科、硕士、博士、高职的权重

    return {
        "year": year,
        "w_b": w_b,
        "w_m": w_m,
        "w_d": w_d,
        "w_v": w_v,
    }


def main():
    # 设置全局字体为中文
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

    # # 合并所有年份的数据
    # combined_df = pd.concat(df.values(), ignore_index=True)

    xls = pd.ExcelFile("1_university_employment_cleaned.xlsx")

    yearly_change = pd.DataFrame(columns=["year", "w_b", "w_m", "w_d", "w_v"])
    for sheet_name in xls.sheet_names:
        sheet = pd.read_excel(xls, sheet_name=sheet_name)

        # 计算各个专业的就业率比例
        ratios = ratio_compute(sheet, int(sheet_name))
        yearly_change = pd.concat(
            [yearly_change, pd.DataFrame(ratios, index=[0])], ignore_index=True
        )
        print(yearly_change)

        # 绘制趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_change["year"], yearly_change["w_b"], marker="o", label="本科")
    plt.plot(yearly_change["year"], yearly_change["w_m"], marker="o", label="硕士")
    plt.plot(yearly_change["year"], yearly_change["w_d"], marker="o", label="博士")
    plt.plot(yearly_change["year"], yearly_change["w_v"], marker="o", label="高职")
    plt.xlabel("年份")
    plt.ylabel("权重")
    plt.title("各学历人数权重随年份变化趋势")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
