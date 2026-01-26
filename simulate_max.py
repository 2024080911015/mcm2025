import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# ==========================================
# 1. 核心工具：Tobit 似然函数 (通用，无需修改)
# ==========================================
def weighted_tobit_log_likelihood(params, X, y, weights):
    beta = params[:-1]
    sigma = params[-1]

    if sigma <= 0: return 1e10  # 惩罚非正 sigma

    mu = np.dot(X, beta)

    censored_mask = (y <= 0)
    uncensored_mask = ~censored_mask

    # 1. 零奖牌部分 (Censored)
    prob_censored = norm.cdf(-mu[censored_mask] / sigma)
    ll_censored = weights[censored_mask] * np.log(np.clip(prob_censored, 1e-15, 1.0))

    # 2. 正奖牌部分 (Uncensored)
    z_score = (y[uncensored_mask] - mu[uncensored_mask]) / sigma
    ll_uncensored = weights[uncensored_mask] * (-np.log(sigma) + norm.logpdf(z_score))

    return - (np.sum(ll_censored) + np.sum(ll_uncensored))


# ==========================================
# 2. 数据准备：计算滞后项与对数
# ==========================================
def load_and_prep_data_advanced():
    # 读取原始数据
    df_early = pd.read_csv('modeling_data_early.csv')
    df_formation = pd.read_csv('modeling_data_formation.csv')
    df_maturity = pd.read_csv('modeling_data_maturity.csv')

    # 设置时间权重
    df_early['weight'] = 0.5722
    df_formation['weight'] = 0.6822
    df_maturity['weight'] = 1.0

    # 1. 合并所有数据 (为了正确计算滞后项，必须先合并)
    full_df = pd.concat([df_early, df_formation, df_maturity], ignore_index=True)

    # 2. 排序：按国家和年份排序，确保 shift 获取的是上一届
    full_df = full_df.sort_values(by=['NOC', 'Year'])

    # 3. 计算滞后项 (Lagged Variables)
    # group by NOC 确保不会把美国的上一届算成中国的
    full_df['Lagged_Total_Share'] = full_df.groupby('NOC')['Medal_Share'].shift(1).fillna(0)
    full_df['Lagged_Gold_Share'] = full_df.groupby('NOC')['Gold_Share'].shift(1).fillna(0)

    # 4. 计算对数特征 (Log Variables)
    # 使用 log(x) (假设人数和项目数至少为1，如果可能为0建议用 np.log1p)
    full_df['Ln_Athletes'] = np.log(full_df['Athlete_Count'])
    full_df['Ln_Events'] = np.log(full_df['Events_Participated_Count'])

    # 5. 确保 Host 是数值型
    full_df['Is_Host'] = full_df['Is_Host'].astype(float)

    return full_df


# ==========================================
# 3. 模型训练 (适配新公式)
# ==========================================
def train_model_advanced(df, target_col, lag_col):
    """
    改进版：包含自动重试机制，专门解决金牌模型不收敛的问题
    """
    y = df[target_col].values
    X_cols = ['Ln_Athletes', 'Ln_Events', 'Is_Host', lag_col]
    X_matrix = df[X_cols].values
    X = np.column_stack([np.ones(len(X_matrix)), X_matrix])
    weights = df['weight'].values

    # === 定义多组初始猜测 ===
    # 猜测 1: 通用型 (适合总奖牌)
    guess_1 = [-0.05, 0.01, 0.01, 0.02, 0.5, 0.05]

    # 猜测 2: 保守型 (适合金牌 - 截距更低，Lag权重更低，Sigma更小)
    guess_2 = [-0.15, 0.005, 0.005, 0.01, 0.3, 0.02]

    # 猜测 3: 激进型 (假设惯性很大)
    guess_3 = [-0.02, 0.02, 0.02, 0.05, 0.8, 0.1]

    guesses = [guess_1, guess_2, guess_3]

    print(f"正在拟合目标 [{target_col}] ...")

    # === 循环尝试不同的初始值 ===
    for i, current_guess in enumerate(guesses):
        try:
            result = minimize(
                weighted_tobit_log_likelihood,
                current_guess,
                args=(X, y, weights),
                method='L-BFGS-B',
                # 稍微放宽精度要求 (ftol)，增加最大迭代次数
                options={'maxiter': 2000, 'ftol': 1e-9},
                bounds=[(None, None)] * 5 + [(1e-6, None)]
            )

            if result.success:
                print(f" -> 尝试第 {i + 1} 组初值: 拟合成功！")
                return result.x
            else:
                # 只有在最后一次尝试失败时才打印错误
                if i == len(guesses) - 1:
                    print(f" -> 所有尝试均失败。最后一次报错: {result.message}")
        except Exception as e:
            print(f" -> 尝试 {i + 1} 发生异常: {e}")

    return None


# ==========================================
# 4. 预测函数 (适配对数与滞后)
# ==========================================
def predict_new_formula(beta_params, sigma, athletes, events, is_host, prev_share):
    """
    根据公式计算：
    mu = b0 + b1*ln(Ath) + b2*ln(Evt) + b3*Host + b4*Prev
    """
    # 1. 转换输入为对数
    ln_ath = np.log(athletes)
    ln_evt = np.log(events)

    # 2. 提取系数
    b0, b1, b2, b3, b4 = beta_params

    # 3. 计算潜在实力 mu
    mu = b0 + (b1 * ln_ath) + (b2 * ln_evt) + (b3 * is_host) + (b4 * prev_share)

    # 4. Tobit 期望值计算
    z = mu / sigma
    expected_share = norm.cdf(z) * mu + sigma * norm.pdf(z)
    prob_winning = norm.cdf(z)

    return expected_share, prob_winning


# ==========================================
# 主程序
# ==========================================
# ==========================================
# 主程序执行 (请替换原代码底部)
# ==========================================
if __name__ == "__main__":
    # 1. 加载并处理数据
    data = load_and_prep_data_advanced()

    # 2. 训练 [总奖牌] 模型
    # 对应公式：M_t = b + b1*lnA + b2*lnE + b3*Host + b4*M_{t-1}
    params_total = train_model_advanced(data, 'Medal_Share', 'Lagged_Total_Share')

    # 3. 训练 [金牌] 模型
    params_gold = train_model_advanced(data, 'Gold_Share', 'Lagged_Gold_Share')

    # --- 结果展示 ---
    print("\n" + "=" * 60)
    print("      NEW LOG-LINEAR TOBIT RESULTS (新公式拟合结果)")
    print("=" * 60)

    # 输出模型 A (总奖牌)
    if params_total is not None:
        p = params_total
        print(f"\n[模型 A: 总奖牌份额 (Total Medal Share)]")
        print(f"  > Constant (beta):         {p[0]:.6f}")
        print(f"  > ln(Athletes) (beta1):    {p[1]:.6f}")
        print(f"  > ln(Events) (beta2):      {p[2]:.6f}")
        print(f"  > Is_Host (beta3):         {p[3]:.6f}   <-- 东道主效应")
        print(f"  > Lagged_Share (beta4):    {p[4]:.6f}   <-- 历史惯性")
        print(f"  > Sigma:                   {p[5]:.6f}")

    # 【补全】输出模型 B (金牌)
    if params_gold is not None:
        p = params_gold
        print(f"\n[模型 B: 金牌份额 (Gold Medal Share)]")
        print(f"  > Constant (beta):         {p[0]:.6f}")
        print(f"  > ln(Athletes) (beta1):    {p[1]:.6f}")
        print(f"  > ln(Events) (beta2):      {p[2]:.6f}")
        print(f"  > Is_Host (beta3):         {p[3]:.6f}   <-- 东道主效应")
        print(f"  > Lagged_Share (beta4):    {p[4]:.6f}   <-- 历史惯性")
        print(f"  > Sigma:                   {p[5]:.6f}")
    else:
        print("\n[模型 B] 未能输出结果 (params_gold is None)")

    # --- 预测演示 (2028 洛杉矶) ---
    print("\n" + "=" * 60)
    print("      2028 PREDICTION DEMO (基于新公式)")
    print("=" * 60)

    # 演示案例
    cases = [
        # 美国：东道主，上届总份额12% (假设金牌份额约为总份额的35% -> 4.2%)
        {"name": "United States (Host)", "ath": 630, "evt": 280, "host": 1, "prev_total": 0.12, "prev_gold": 0.042},
        # 中国：非东道主，上届总份额9% (假设金牌份额较高 -> 3.5%)
        {"name": "China", "ath": 420, "evt": 240, "host": 0, "prev_total": 0.09, "prev_gold": 0.035},
        # 发展中国家
        {"name": "Developing Country", "ath": 100, "evt": 40, "host": 0, "prev_total": 0.005, "prev_gold": 0.001}
    ]

    for c in cases:
        print(f"\n国家: {c['name']}")

        # 预测总奖牌
        if params_total is not None:
            share_t, _ = predict_new_formula(
                params_total[:-1], params_total[-1],
                c['ath'], c['evt'], c['host'], c['prev_total']
            )
            print(f"  - 预期总奖牌份额: {share_t:.4%}")

        # 【补全】预测金牌
        if params_gold is not None:
            share_g, prob_g = predict_new_formula(
                params_gold[:-1], params_gold[-1],
                c['ath'], c['evt'], c['host'], c['prev_gold']
            )
            print(f"  - 预期金牌份额:   {share_g:.4%} (拿金牌概率: {prob_g:.2%})")