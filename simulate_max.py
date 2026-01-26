import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

# 忽略除零等微小的数值警告
warnings.filterwarnings('ignore')


# ==========================================
# 1. 核心工具：Tobit 似然函数
# ==========================================
def weighted_tobit_log_likelihood(params, X, y, weights):
    """
    计算加权 Tobit 模型的对数似然值
    """
    beta = params[:-1]
    sigma = params[-1]

    # 惩罚无效的 sigma
    if sigma <= 1e-6: return 1e10

    mu = np.dot(X, beta)

    censored_mask = (y <= 0)
    uncensored_mask = ~censored_mask

    # 1. 零奖牌部分 (Censored at 0)
    # Prob(y* <= 0) = Phi(-mu / sigma)
    prob_censored = norm.cdf(-mu[censored_mask] / sigma)
    ll_censored = weights[censored_mask] * np.log(np.clip(prob_censored, 1e-15, 1.0))

    # 2. 正奖牌部分 (Uncensored)
    # PDF = (1/sigma) * phi((y - mu) / sigma)
    z_score = (y[uncensored_mask] - mu[uncensored_mask]) / sigma
    ll_uncensored = weights[uncensored_mask] * (-np.log(sigma) + norm.logpdf(z_score))

    return - (np.sum(ll_censored) + np.sum(ll_uncensored))


# ==========================================
# 2. 统计推断工具：SE, CI, Covariance
# ==========================================
def compute_statistics(params, X, y, weights, epsilon=1e-5):
    """
    计算统计量：标准误 (SE)、P值、协方差矩阵、参数置信区间
    """
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    f0 = weighted_tobit_log_likelihood(params, X, y, weights)

    # --- 数值微分计算 Hessian (二阶导矩阵) ---
    for i in range(n_params):
        for j in range(i, n_params):
            ei = np.zeros(n_params)
            ej = np.zeros(n_params)
            ei[i] = 1.0
            ej[j] = 1.0

            # 自适应步长
            h_i = epsilon * max(abs(params[i]), 1.0)
            h_j = epsilon * max(abs(params[j]), 1.0)

            if i == j:
                f_plus = weighted_tobit_log_likelihood(params + h_i * ei, X, y, weights)
                f_minus = weighted_tobit_log_likelihood(params - h_i * ei, X, y, weights)
                hessian[i, i] = (f_plus - 2 * f0 + f_minus) / (h_i ** 2)
            else:
                f_pp = weighted_tobit_log_likelihood(params + h_i * ei + h_j * ej, X, y, weights)
                f_pm = weighted_tobit_log_likelihood(params + h_i * ei - h_j * ej, X, y, weights)
                f_mp = weighted_tobit_log_likelihood(params - h_i * ei + h_j * ej, X, y, weights)
                f_mm = weighted_tobit_log_likelihood(params - h_i * ei - h_j * ej, X, y, weights)
                val = (f_pp - f_pm - f_mp + f_mm) / (4 * h_i * h_j)
                hessian[i, j] = val
                hessian[j, i] = val

    # --- 计算协方差矩阵与标准误 ---
    try:
        cov_matrix = np.linalg.inv(hessian)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        print("警告: Hessian 矩阵不可逆，无法计算标准误。")
        return None, None, None, None

    # --- 计算 P值 (双尾检验) ---
    t_stats = params / se
    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

    # --- 计算 95% 置信区间 ---
    z_crit = 1.96
    ci_lower = params - z_crit * se
    ci_upper = params + z_crit * se

    return se, p_values, cov_matrix, (ci_lower, ci_upper)


# ==========================================
# 3. 预测函数 (带置信区间)
# ==========================================
def predict_with_ci(params, cov_matrix, athletes, events, is_host, prev_share):
    """
    输入新数据，输出预测值及其 95% 置信区间
    """
    beta = params[:-1]
    sigma = params[-1]

    # 构造输入向量 [1, ln_ath, ln_evt, host, lag]
    x_in = np.array([1, np.log(athletes), np.log(events), is_host, prev_share])

    # 1. 计算潜变量 (Latent Variable) mu = X * beta
    mu = np.dot(x_in, beta)

    # 2. 计算潜变量的方差 (Delta Method)
    # Var(mu) = x^T * Cov(beta) * x
    # 注意：cov_matrix 包含 sigma，我们只需要 beta 部分 (前5行5列)
    beta_cov = cov_matrix[:-1, :-1]
    mu_var = np.dot(x_in.T, np.dot(beta_cov, x_in))
    mu_se = np.sqrt(mu_var)

    # 3. 定义 Tobit 期望转化函数
    def tobit_expected(m, s):
        z = m / s
        return norm.cdf(z) * m + s * norm.pdf(z)

    # 4. 计算点估计
    expected_share = tobit_expected(mu, sigma)

    # 5. 计算置信区间 (基于潜变量的波动映射)
    z_crit = 1.96
    mu_lower = mu - z_crit * mu_se
    mu_upper = mu + z_crit * mu_se

    ci_low = tobit_expected(mu_lower, sigma)
    ci_high = tobit_expected(mu_upper, sigma)

    return expected_share, ci_low, ci_high


# ==========================================
# 4. 数据加载与模型训练
# ==========================================
def load_and_prep_data_advanced():
    try:
        df_early = pd.read_csv('modeling_data_early.csv')
        df_formation = pd.read_csv('modeling_data_formation.csv')
        df_maturity = pd.read_csv('modeling_data_maturity.csv')
    except FileNotFoundError:
        print("错误: 找不到 CSV 文件。请确保 modeling_data_*.csv 在当前目录。")
        return None

    df_early['weight'] = 0.5722
    df_formation['weight'] = 0.6822
    df_maturity['weight'] = 1.0

    full_df = pd.concat([df_early, df_formation, df_maturity], ignore_index=True)
    full_df = full_df.sort_values(by=['NOC', 'Year'])

    # 特征工程
    full_df['Lagged_Total_Share'] = full_df.groupby('NOC')['Medal_Share'].shift(1).fillna(0)
    full_df['Lagged_Gold_Share'] = full_df.groupby('NOC')['Gold_Share'].shift(1).fillna(0)
    full_df['Ln_Athletes'] = np.log(full_df['Athlete_Count'])
    full_df['Ln_Events'] = np.log(full_df['Events_Participated_Count'])
    full_df['Is_Host'] = full_df['Is_Host'].astype(float)

    return full_df


def train_model_wrapper(df, target_col, lag_col):
    """
    训练模型并返回参数和用于计算统计量的矩阵
    """
    y = df[target_col].values
    X_cols = ['Ln_Athletes', 'Ln_Events', 'Is_Host', lag_col]

    # 添加截距项
    X_matrix = df[X_cols].values
    X = np.column_stack([np.ones(len(X_matrix)), X_matrix])
    weights = df['weight'].values

    # 多组初始猜测，防止局部最优
    guesses = [
        [-0.05, 0.01, 0.01, 0.02, 0.5, 0.05],
        [-0.15, 0.005, 0.005, 0.01, 0.3, 0.02],
        [-0.02, 0.02, 0.02, 0.05, 0.8, 0.1]
    ]

    print(f"正在拟合目标 [{target_col}] ...")

    final_res = None

    for i, guess in enumerate(guesses):
        try:
            res = minimize(
                weighted_tobit_log_likelihood,
                guess,
                args=(X, y, weights),
                method='L-BFGS-B',
                bounds=[(None, None)] * 5 + [(1e-6, None)],  # Sigma 必须 > 0
                options={'maxiter': 3000, 'ftol': 1e-9}
            )
            if res.success:
                final_res = res
                print(f" -> 第 {i + 1} 组初值收敛成功 (LogL: {-res.fun:.2f})")
                break
        except Exception as e:
            continue

    if final_res and final_res.success:
        return final_res.x, X, y, weights
    else:
        print(f" -> [{target_col}] 拟合失败。")
        return None, None, None, None


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    data = load_and_prep_data_advanced()

    if data is not None:
        # 定义要运行的模型配置
        models_config = [
            {
                "name": "Total Medal Share",
                "target": "Medal_Share",
                "lag": "Lagged_Total_Share"
            },
            {
                "name": "Gold Medal Share",
                "target": "Gold_Share",
                "lag": "Lagged_Gold_Share"
            }
        ]

        # 存储训练好的参数用于最后的演示
        trained_models = {}

        print("\n" + "=" * 80)
        print("                  TOBIT MODEL RESULTS (WITH CONFIDENCE INTERVALS)")
        print("=" * 80)

        for config in models_config:
            target = config['target']
            model_name = config['name']

            # A. 训练模型
            params, X_train, y_train, w_train = train_model_wrapper(data, target, config['lag'])

            if params is not None:
                # B. 计算统计量
                se, p_vals, cov_matrix, (ci_low, ci_high) = compute_statistics(params, X_train, y_train, w_train)

                # 保存用于后续预测
                trained_models[target] = {
                    "params": params,
                    "cov": cov_matrix,
                    "lag_col": config['lag']
                }

                # C. 打印结果表
                if se is not None:
                    print(f"\n[Model: {model_name}]")
                    print(f"{'Variable':<20} {'Coeff':<10} {'Std.Err':<10} {'P-Value':<10} {'95% CI':<20}")
                    print("-" * 75)

                    var_names = ["Intercept", "ln(Athletes)", "ln(Events)", "Is_Host", "Lagged_Share", "Sigma"]

                    for i, name in enumerate(var_names):
                        # 星号标记显著性
                        sig_star = "***" if p_vals[i] < 0.01 else "**" if p_vals[i] < 0.05 else "*" if p_vals[
                                                                                                           i] < 0.1 else ""

                        print(
                            f"{name:<20} {params[i]:<10.5f} {se[i]:<10.5f} {p_vals[i]:<10.4f} [{ci_low[i]:.4f}, {ci_high[i]:.4f}] {sig_star}")
                    print("-" * 75)
                else:
                    print(f"\n[Model: {model_name}] 统计量计算失败 (Hessian不可逆)")

        # ==========================================
        # 6. 2028 洛杉矶奥运会 预测演示 (带区间)
        # ==========================================
        print("\n" + "=" * 80)
        print("                  2028 PREDICTION DEMO (WITH 95% CI)")
        print("=" * 80)

        # 演示案例数据
        cases = [
            {"name": "USA (Host)", "ath": 630, "evt": 300, "host": 1,
             "prev_total": 0.12, "prev_gold": 0.045},  # 假设数据
            {"name": "China", "ath": 420, "evt": 240, "host": 0,
             "prev_total": 0.09, "prev_gold": 0.038},
            {"name": "Small Nation", "ath": 50, "evt": 20, "host": 0,
             "prev_total": 0.00, "prev_gold": 0.00}
        ]

        for c in cases:
            print(f"\n国家: {c['name']}")

            # 预测总奖牌
            if "Medal_Share" in trained_models:
                m = trained_models["Medal_Share"]
                est, low, high = predict_with_ci(
                    m['params'], m['cov'],
                    c['ath'], c['evt'], c['host'], c['prev_total']
                )
                print(f"  > 总奖牌预测: {est:.4%} (95% CI: {low:.4%} - {high:.4%})")

            # 预测金牌
            if "Gold_Share" in trained_models:
                m = trained_models["Gold_Share"]
                est, low, high = predict_with_ci(
                    m['params'], m['cov'],
                    c['ath'], c['evt'], c['host'], c['prev_gold']
                )
                print(f"  > 金牌预测:   {est:.4%} (95% CI: {low:.4%} - {high:.4%})")