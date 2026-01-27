import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# ==========================================
# 1. 准备数据与训练模型
# ==========================================
df_all = pd.read_csv('modeling_data_all.csv')
df_all = df_all.sort_values(['NOC', 'Year'])
df_all['Lagged_Medals'] = df_all.groupby('NOC')['Total_Medals'].shift(1).fillna(0)

train_data = df_all[df_all['Year'] > 1896].dropna(
    subset=['Total_Medals', 'Lagged_Medals', 'Is_Host', 'Athlete_Count', 'Events_Participated_Count'])
y_train = train_data['Total_Medals']
X_train = sm.add_constant(train_data[['Lagged_Medals', 'Is_Host']])
Z_train = sm.add_constant(train_data[['Athlete_Count', 'Events_Participated_Count']])

print("Retraining ZIP model...")
zip_model_res = sm.ZeroInflatedPoisson(endog=y_train, exog=X_train, exog_infl=Z_train, inflation='logit').fit(
    method='bfgs', maxiter=1000, disp=0)
params = zip_model_res.params
gamma_const = params['inflate_const']
gamma_athlete = params['inflate_Athlete_Count']
gamma_events = params['inflate_Events_Participated_Count']

# ==========================================
# 2. 准备绘图数据 (Top 7 Candidates)
# ==========================================
df_2028_pred = pd.read_csv('prediction_2028_final.csv')
noc_medal_history = df_all.groupby('NOC')['Total_Medals'].sum()
never_medalists = noc_medal_history[noc_medal_history == 0].index.tolist()
candidates_2028 = df_2028_pred[df_2028_pred['NOC'].isin(never_medalists)].copy()

candidates_2028['Logit_Score'] = gamma_const + gamma_athlete * candidates_2028['Athlete_Count'] + gamma_events * \
                                 candidates_2028['Events_Participated_Count']
candidates_2028['Pi'] = 1 / (1 + np.exp(-candidates_2028['Logit_Score']))

top_candidates = candidates_2028.sort_values('Pi', ascending=True).head(7)
xs = top_candidates['Athlete_Count'].values
ys = top_candidates['Events_Participated_Count'].values
zs = top_candidates['Pi'].values
labels = top_candidates['NOC'].values

# ==========================================
# 3. 绘制 3D 投影图 (Legend Version)
# ==========================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# A. 绘制数学曲面
x_range = np.linspace(0, 50, 50)
y_range = np.linspace(0, 20, 50)
X, Y = np.meshgrid(x_range, y_range)
Score = gamma_const + gamma_athlete * X + gamma_events * Y
Z_surf = 1 / (1 + np.exp(-Score))
surf = ax.plot_surface(X, Y, Z_surf, cmap='coolwarm_r', alpha=0.2, edgecolor='none', rstride=1, cstride=1)

# B. 坐标抖动 (Jittering)
np.random.seed(42)
jitter_amount = 0.8
xs_jitter = xs + np.random.uniform(-jitter_amount, jitter_amount, size=len(xs))
ys_jitter = ys + np.random.uniform(-jitter_amount, jitter_amount, size=len(ys))

# C. 绘制点线 (使用7种不同颜色)
cmap = cm.get_cmap('tab10')  # 使用高区分度色板
colors = [cmap(i) for i in range(7)]
scatter_proxies = []  # 用于生成图例

for i, (x_j, y_j, z, label) in enumerate(zip(xs_jitter, ys_jitter, zs, labels)):
    color = colors[i]

    # 底部投影点
    ax.scatter(x_j, y_j, 0, color=color, s=100, edgecolors='black', alpha=0.9, depthshade=False)

    # 虚线连接
    ax.plot([x_j, x_j], [y_j, y_j], [0, z], color=color, linestyle='--', linewidth=2.0, alpha=0.8)

    # 顶部悬浮点
    sc = ax.scatter(x_j, y_j, z, color=color, s=60, edgecolors='white', alpha=1.0)

    # 收集图例句柄
    scatter_proxies.append(sc)

# D. 装饰
ax.set_xlabel('team_size', fontsize=12, labelpad=10)
ax.set_ylabel('Sports_count', fontsize=12, labelpad=10)
ax.set_zlabel('πi', fontsize=12, labelpad=10)
ax.set_title('Top 7 Candidates: Breaking the Barrier\n(Dashed Lines = Projected Probability)', fontsize=16,
             weight='bold')

ax.set_zlim(-0.1, 1.0)
ax.view_init(elev=25, azim=130)

# E. 添加图例 (放在右侧)
# E. 添加图例 (放在右侧)
# 将 bbox_to_anchor 中的第一个数字从 1.05 改大，比如 1.20 或 1.25
ax.legend(scatter_proxies, labels, title="Country (NOC)", loc="center left", bbox_to_anchor=(1.20, 0.5), fontsize=12,
          title_fontsize=13)

plt.tight_layout()
plt.savefig('3d_breakthrough_top7_legend.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存: 3d_breakthrough_top7_legend.png")
plt.show()



