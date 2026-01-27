import pandas as pd
import statsmodels.api as sm
import numpy as np

# ==========================================
# Step 1: 数据加载与预处理 (Data Preparation)
# ==========================================
print("1. Loading and preparing data...")

# 读取历史数据和2028预测数据
# 请确保文件名与你的本地文件一致
try:
    df_all = pd.read_csv('modeling_data_all.csv')
    df_2028_pred = pd.read_csv('prediction_2028_final.csv')
except FileNotFoundError:
    print("Error: CSV files not found. Please check file paths.")
    exit()

# 按国家和年份排序，确保滞后项计算正确
df_all = df_all.sort_values(['NOC', 'Year'])

# 特征工程：计算上一届奖牌数 (Lagged_Medals)
# 如果是第一年参赛，fillna(0) 设为 0
df_all['Lagged_Medals'] = df_all.groupby('NOC')['Total_Medals'].shift(1).fillna(0)

# 识别“从未拿牌国家” (Never Medalists)
# 只要历史总和为 0，就算作 2028 的潜在突破对象
noc_medal_history = df_all.groupby('NOC')['Total_Medals'].sum()
never_medalists = noc_medal_history[noc_medal_history == 0].index.tolist()
print(f"-> Found {len(never_medalists)} nations that have never won a medal.")

# 准备训练集 (Training Set)
# 排除 1896 年 (没有上一届数据) 并去除缺失值
train_data = df_all[df_all['Year'] > 1896].copy()
train_data = train_data.dropna(
    subset=['Total_Medals', 'Lagged_Medals', 'Is_Host', 'Athlete_Count', 'Events_Participated_Count'])

# ==========================================
# Step 2: 训练 ZIP 模型 (Model Training)
# ==========================================
print("\n2. Training Zero-Inflated Poisson (ZIP) Model...")

# 定义 Y (因变量)
y_train = train_data['Total_Medals']

# 定义 X (Poisson 部分 - 决定拿几块)
# 包含：常数项、上一届奖牌、是否东道主
X_train = train_data[['Lagged_Medals', 'Is_Host']]
X_train = sm.add_constant(X_train)

# 定义 Z (Logit 部分 - 决定是否绝缘)
# 包含：常数项、运动员人数、参赛项目数
Z_train = train_data[['Athlete_Count', 'Events_Participated_Count']]
Z_train = sm.add_constant(Z_train)

# 训练模型 (使用 BFGS 算法以保证收敛)
try:
    zip_model_res = sm.ZeroInflatedPoisson(
        endog=y_train,
        exog=X_train,
        exog_infl=Z_train,
        inflation='logit'
    ).fit(method='bfgs', maxiter=1000, disp=0)

    print("-> Model training successful!")
    print("\n--- Model Parameters ---")
    print(zip_model_res.params)
except Exception as e:
    print(f"Error in training: {e}")
    exit()

# ==========================================
# Step 3: 预测 2028 (Prediction)
# ==========================================
print("\n3. Predicting for 2028 Candidates...")

# 筛选出 2028 年参赛且从未拿过牌的国家
candidates_2028 = df_2028_pred[df_2028_pred['NOC'].isin(never_medalists)].copy()

# 准备预测变量
# 对于从未拿牌的国家，Lagged_Medals 必然是 0
# 假设 Is_Host 为 0 (除非有些从未拿牌的国家突然办奥运，这不可能)
candidates_2028['Lagged_Medals'] = 0
candidates_2028['Is_Host'] = 0

# 提取模型参数
params = zip_model_res.params
gamma_const = params['inflate_const']
gamma_athlete = params['inflate_Athlete_Count']
gamma_events = params['inflate_Events_Participated_Count']
beta_const = params['const']
# beta_lagged 和 beta_host 在此场景下乘以 0，所以只用 beta_const

# --- 核心计算逻辑 ---

# 1. 计算 Logit Score (线性部分)
# Score = gamma0 + gamma1*人数 + gamma2*项目数
Z1 = candidates_2028['Athlete_Count']
Z2 = candidates_2028['Events_Participated_Count']
logit_score = gamma_const + (gamma_athlete * Z1) + (gamma_events * Z2)

# 2. 计算 pi (绝缘概率 - Structural Zero Probability)
# pi = 1 / (1 + e^-Score)
pi = 1 / (1 + np.exp(-logit_score))

# 3. 计算 Lambda (期望奖牌数 - Poisson Mean)
# lambda = exp(beta0)  <-- 因为 Lagged=0, Host=0
lambda_val = np.exp(beta_const)

# 4. 计算拿 0 牌的总概率 P(Y=0)
# P(Y=0) = 绝缘的概率 + (不绝缘但运气不好拿0个的概率)
prob_zero = pi + (1 - pi) * np.exp(-lambda_val)

# 5. 计算突破概率 P(Y>=1)
prob_breakthrough = 1 - prob_zero

# 将结果存入表格
candidates_2028['Breakthrough_Prob'] = prob_breakthrough

# ==========================================
# Step 4: 结果展示 (Results)
# ==========================================
print("\n--- 2028 Prediction Results ---")

# 1. 期望突破国家数量 (Expected Count)
expected_new_winners = prob_breakthrough.sum()
print(f"Expected number of First-Time Winners: {expected_new_winners:.4f}")

# 2. 排名靠前的“黑马” (Top Candidates)
top_candidates = candidates_2028.sort_values('Breakthrough_Prob', ascending=False)
print("\nTop 10 Candidates for First Medal:")
print(
    top_candidates[['NOC', 'Country_Name', 'Breakthrough_Prob', 'Athlete_Count', 'Events_Participated_Count']].head(10))

# 3. 保存结果
top_candidates.to_csv("2028_breakthrough_predictions.csv", index=False)
print("\nResults saved to 2028_breakthrough_predictions.csv")