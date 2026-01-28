import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

# =========================
# 1. 数据处理 (与之前保持一致)
# =========================
try:
    df = pd.read_csv('country_sport_CR_RCA.csv')
except FileNotFoundError:
    try:
        df_raw = pd.read_csv('summerOly_athletes.csv')
        df_medals = df_raw[df_raw['Medal'].isin(['Gold', 'Silver', 'Bronze'])].copy()
        df_medals = df_medals[df_medals['Year'] >= 2000]
        df_medals = df_medals.drop_duplicates(subset=['Year', 'Event', 'Medal', 'NOC'])

        medals_cs = df_medals.groupby(['NOC', 'Sport']).size().reset_index(name='Medals_cs')
        total_c = medals_cs.groupby('NOC')['Medals_cs'].sum().reset_index(name='TotalMedals_c')
        total_s = medals_cs.groupby('Sport')['Medals_cs'].sum().reset_index(name='TotalMedals_s')
        total_world = total_c['TotalMedals_c'].sum()

        df = medals_cs.merge(total_c, on='NOC').merge(total_s, on='Sport')
        df['CR_c_s'] = df['Medals_cs'] / df['TotalMedals_c']
        df['Share_s_World'] = df['TotalMedals_s'] / total_world
        df['RCA_c_s'] = df['CR_c_s'] / df['Share_s_World']
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()

# 筛选前 25 个主要国家
top_countries = (
    df[['NOC', 'TotalMedals_c']]
    .drop_duplicates()
    .sort_values('TotalMedals_c', ascending=False)
    .head(25)['NOC']
    .tolist()
)
df_sub = df[df['NOC'].isin(top_countries)].copy()

# 构建矩阵
rca_matrix = df_sub.pivot(index='NOC', columns='Sport', values='RCA_c_s').fillna(0)
cr_matrix = df_sub.pivot(index='NOC', columns='Sport', values='CR_c_s').fillna(0)
medals_matrix = df_sub.pivot(index='NOC', columns='Sport', values='Medals_cs').fillna(0)

# 对齐索引
common_index = rca_matrix.index
common_columns = rca_matrix.columns
cr_matrix = cr_matrix.loc[common_index, common_columns]
medals_matrix = medals_matrix.loc[common_index, common_columns]

# 聚类排序
row_linkage = linkage(rca_matrix, method='ward', metric='euclidean')
row_order = rca_matrix.index[leaves_list(row_linkage)]

col_linkage = linkage(rca_matrix.T, method='ward', metric='euclidean')
col_order = rca_matrix.columns[leaves_list(col_linkage)]

rca_matrix = rca_matrix.loc[row_order, col_order]
cr_matrix = cr_matrix.loc[row_order, col_order]
medals_matrix = medals_matrix.loc[row_order, col_order]

# Top-K 掩码
K = 5
mask_top_k = pd.DataFrame(False, index=rca_matrix.index, columns=rca_matrix.columns)
for country in rca_matrix.index:
    top_sports = medals_matrix.loc[country].nlargest(K).index
    valid_top = [s for s in top_sports if medals_matrix.loc[country, s] > 0]
    mask_top_k.loc[country, valid_top] = True

# =========================
# 2. 绘图
# =========================
sns.set_theme(style="white", font_scale=1.1)
fig, ax = plt.subplots(figsize=(24, 14), dpi=300)

vmax = np.nanpercentile(rca_matrix.values, 95)
vmax = max(1.5, min(vmax, 4.0))

# --- Heatmap (RCA) ---
hm = sns.heatmap(
    rca_matrix,
    cmap="YlOrRd",
    vmin=0,
    vmax=vmax,
    annot=False,
    linewidths=0.5,
    linecolor="#444444",
    cbar_kws={
        'label': 'RCA',
        'shrink': 0.4,
        'anchor': (0.0, 1.0),
        'pad': 0.02
    },
    ax=ax
)

# --- Bubbles (BCR) ---
xs, ys, sizes = [], [], []
rows = rca_matrix.index
cols = rca_matrix.columns

for i, country in enumerate(rows):
    for j, sport in enumerate(cols):
        if mask_top_k.loc[country, sport]:
            cr_val = cr_matrix.loc[country, sport]
            if cr_val > 0:
                xs.append(j + 0.5)
                ys.append(i + 0.5)
                sizes.append(cr_val * 900)

ax.scatter(
    xs, ys,
    s=sizes,
    facecolors='none',
    edgecolors='#333333',
    linewidths=1.8,
    alpha=0.9,
    zorder=10
)

# --- 标签设置 ---
ax.set_title("")
ax.set_xlabel("Sport", fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel("Country", fontsize=14, labelpad=15, fontweight='bold')

# --- 关键修改：X轴文字斜着写并对齐 ---
# rotation=45: 旋转45度
# ha='right': 右对齐 (文字末端对齐刻度线，视觉上正好指着格子)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

# --- 图例设置 ---
# RCA 颜色条
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.set_label("RCA", fontsize=12, fontweight='bold', labelpad=10)

# BCR 气泡图例
legend_sizes = [0.10, 0.30, 0.50]
legend_handles = [
    plt.scatter([], [], s=s * 900, facecolors='none', edgecolors='#333333', linewidths=1.8)
    for s in legend_sizes
]

leg = ax.legend(
    legend_handles,
    [f'{s * 100:.0f}%' for s in legend_sizes],
    title="BCR",
    title_fontsize=12,
    fontsize=10,
    loc='lower left',
    bbox_to_anchor=(1.03, 0.0),
    frameon=True,
    edgecolor='#cccccc',
    labelspacing=1.5,
    borderpad=1.0
)
leg.get_title().set_fontweight('bold')

plt.subplots_adjust(right=0.85, bottom=0.15)
plt.savefig('country_sport_matrix_final_v2.png', bbox_inches='tight', dpi=600)
plt.show()