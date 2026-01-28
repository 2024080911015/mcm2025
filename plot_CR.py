import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

# =========================
# 1. 数据读取与处理
# =========================
try:
    df = pd.read_csv('country_sport_CR_RCA.csv')
except FileNotFoundError:
    # 如果找不到衍生文件，则从原始数据重新计算
    try:
        df_raw = pd.read_csv('summerOly_athletes.csv')
        df_medals = df_raw[df_raw['Medal'].isin(['Gold', 'Silver', 'Bronze'])].copy()
        df_medals = df_medals[df_medals['Year'] >= 2000]  # 聚焦现代奥运
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

# 聚类排序 (Clustering)
row_linkage = linkage(rca_matrix, method='ward', metric='euclidean')
row_order = rca_matrix.index[leaves_list(row_linkage)]

col_linkage = linkage(rca_matrix.T, method='ward', metric='euclidean')
col_order = rca_matrix.columns[leaves_list(col_linkage)]

rca_matrix = rca_matrix.loc[row_order, col_order]
cr_matrix = cr_matrix.loc[row_order, col_order]
medals_matrix = medals_matrix.loc[row_order, col_order]

# 计算 Top-K 掩码 (仅显示每国前 5 强项的气泡)
K = 5
mask_top_k = pd.DataFrame(False, index=rca_matrix.index, columns=rca_matrix.columns)
for country in rca_matrix.index:
    top_sports = medals_matrix.loc[country].nlargest(K).index
    valid_top = [s for s in top_sports if medals_matrix.loc[country, s] > 0]
    mask_top_k.loc[country, valid_top] = True

# =========================
# 2. 绘图 (样式定制)
# =========================
sns.set_theme(style="white", font_scale=1.1)
fig, ax = plt.subplots(figsize=(24, 14), dpi=300)

# 设置颜色映射上限，避免极端值影响
vmax = np.nanpercentile(rca_matrix.values, 95)
vmax = max(1.5, min(vmax, 4.0))

# --- A. 绘制热力图 (RCA) ---
# cmap='YlOrRd': 浅黄到红色
# linecolor='#444444': 深灰色网格线
# cbar_kws: 调整颜色条大小和位置，实现“份额各半”
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
        'shrink': 0.4,  # 缩短颜色条，为下方气泡图例腾出空间
        'anchor': (0.0, 1.0),  # 锚定在上方
        'pad': 0.02
    },
    ax=ax
)

# --- B. 绘制气泡 (BCR) ---
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
                sizes.append(cr_val * 900)  # 气泡大小系数

ax.scatter(
    xs, ys,
    s=sizes,
    facecolors='none',
    edgecolors='#333333',  # 深灰色描边，保证高对比度
    linewidths=1.8,
    alpha=0.9,
    zorder=10
)

# --- C. 标签与对齐 ---
ax.set_title("")
ax.set_xlabel("Sport", fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel("Country", fontsize=14, labelpad=15, fontweight='bold')

# X轴标签旋转90度并居中对齐，确保文字正对格子
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

# --- D. 图例调整 (各占一半) ---
# 1. 颜色条调整 (RCA)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.set_label("RCA", fontsize=12, fontweight='bold', labelpad=10)

# 2. 气泡图例调整 (BCR)
legend_sizes = [0.10, 0.30, 0.50]
legend_handles = [
    plt.scatter([], [], s=s * 900, facecolors='none', edgecolors='#333333', linewidths=1.8)
    for s in legend_sizes
]

# 将 BCR 图例放置在右下角 (1.03, 0.0)
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

# 调整边距，防止图例被裁剪
plt.subplots_adjust(right=0.85, bottom=0.15)

# 保存图片
plt.savefig('country_sport_matrix_final.png', bbox_inches='tight', dpi=500)
plt.show()