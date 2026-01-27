import pandas as pd
import numpy as np
import statsmodels.api as sm

# ==========================================
# 1. 数据准备与清洗
# ==========================================
data_all = pd.read_csv("modeling_data_maturity.csv")

# --- 处理 2024 年数据 (Target Year) ---
df_2024 = data_all[data_all['Year'] == 2024].copy()

# 确保 Is_Host 变量准确 (2024年东道主是法国 FRA)
# 如果数据中已有 Is_Host 列且准确，可以直接用；这里手动计算作为双重保险
df_2024['Is_Host'] = np.where(df_2024['NOC'] == 'FRA', 1, 0)

# --- 处理 2020 年数据 (Lagged Variable) ---
df_2020 = data_all[data_all['Year'] == 2020][['NOC', 'Total_Medals']].copy()
df_2020.rename(columns={'Total_Medals': 'Lagged_Medals_2020'}, inplace=True)

# --- 合并数据 ---
# 保留 2024 年的所有参赛国家 (Left Join)
df_merged = pd.merge(df_2024, df_2020, on='NOC', how='left')

# 填充缺失值：如果 2020 年没参赛，则滞后奖牌数为 0
df_merged['Lagged_Medals_2020'] = df_merged['Lagged_Medals_2020'].fillna(0)

# --- 关键修改：变量映射 ---
# 建议使用 'Athlete_Count' 代表团队规模 (team_size)，而非 'Events_Participated_Count'
df_merged.rename(columns={
    'Athlete_Count': 'team_size_2024',          # <--- 修改点：使用运动员人数
    'Sports_Participated_Count': 'sports_count_2024',
    'Total_Medals': 'medal_count_2024'
}, inplace=True)

# ==========================================
# 2. 模型构建与拟合
# ==========================================
df = df_merged.copy()

# 定义变量
y = df['medal_count_2024']

# Poisson 部分 (预测奖牌数量)
X_vars = ['Lagged_Medals_2020', 'Is_Host']
X_beta = sm.add_constant(df[X_vars])

# Zero Inflation 部分 (预测是否得0牌)
Z_vars = ['team_size_2024', 'sports_count_2024']
Z_gamma = sm.add_constant(df[Z_vars])

print("正在拟合 Zero-Inflated Poisson 模型...")

# 实例化模型
model = sm.ZeroInflatedPoisson(endog=y, exog=X_beta, exog_infl=Z_gamma, inflation='logit')

# 求解 (优先使用 BFGS 算法，更稳定)
try:
    results = model.fit(method='bfgs', maxiter=5000, disp=False)
    print("\n模型拟合成功！摘要如下：")
    print(results.summary())
except Exception as e:
    print(f"\nBFGS 拟合失败 ({e})，尝试 Newton-Raphson 算法...")
    try:
        results = model.fit(method='newton', maxiter=5000, disp=False)
        print("\n模型拟合成功！摘要如下：")
        print(results.summary())
    except Exception as e2:
        print(f"\n模型拟合最终失败: {e2}")