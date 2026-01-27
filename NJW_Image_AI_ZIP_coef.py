import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 1. 准备 Top 25 数据 (Expanded Dataset)
# ==========================================
# 数据来源：基于 ZIP 模型对 2028 年所有未拿牌国家的完整预测
data = {
    'Country': [
        'Angola', 'Guinea', 'Mali', 'Lebanon', 'South Sudan',
        'El Salvador', 'Palestine', 'Guam', 'Rwanda', 'Gambia',
        'Nicaragua', 'Madagascar', 'Papua New Guinea', 'Liberia', 'Nepal',
        'Aruba', 'Libya', 'Vanuatu', 'DR Congo', 'Monaco',
        'Guinea Bissau', 'Bosnia & Herz.', 'Benin', 'Malta', 'Antigua & Barb.'
    ],
    'Probability': [
        0.277, 0.251, 0.234, 0.146, 0.142,
        0.140, 0.135, 0.135, 0.130, 0.130,
        0.124, 0.124, 0.124, 0.120, 0.120,
        0.119, 0.115, 0.115, 0.115, 0.115,
        0.115, 0.110, 0.106, 0.106, 0.106
    ],
    'Athletes': [
        25, 25, 24, 9, 14,
        8, 8, 7, 7, 7,
        7, 7, 7, 8, 7,
        6, 6, 6, 6, 6,
        6, 5, 5, 5, 5
    ],
    'Events': [
        10, 7, 6, 9, 3,
        9, 8, 9, 8, 8,
        7, 7, 7, 5, 6,
        7, 6, 6, 6, 6,
        6, 6, 5, 5, 5
    ]
}
df = pd.DataFrame(data)

# ==========================================
# 2. 设置绘图风格
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.0) # 稍微调小字体以适应更多数据
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 3. 图表 A: 升级版突破矩阵 (Bubble Chart)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8)) # 画布加大

# 绘制散点
scatter = sns.scatterplot(
    data=df,
    x='Athletes',
    y='Events',
    size='Probability',
    sizes=(100, 1000), # 气泡大小范围
    hue='Probability',
    palette='viridis',
    alpha=0.7,
    edgecolor='black',
    linewidth=1.0,
    ax=ax
)

# 智能添加标签 (避免重叠，只标注 Top 15 或特殊的)
# 这里为了信息量，标注所有点，但稍微错开位置
for i in range(df.shape[0]):
    # 简单的避让逻辑：奇数行向上偏，偶数行向下偏
    y_offset = 0.3 if i % 2 == 0 else -0.3
    ax.text(
        df.Athletes[i] + 0.3,
        df.Events[i] + y_offset,
        df.Country[i],
        fontsize=9,
        color='#333333',
        alpha=0.9
    )

ax.set_title("The Breakthrough Matrix (Top 25 Candidates)\nResource Input vs. Probability", fontsize=16, weight='bold', pad=20)
ax.set_xlabel("Delegation Size (Number of Athletes)", fontsize=12)
ax.set_ylabel("Participation Breadth (Number of Events)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

# 调整图例位置
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., title="Prob.")
plt.tight_layout()
plt.savefig('Chart1_Matrix_Top25.png', dpi=300)
print("✅ 图表 1 (Top 25 矩阵) 已保存")

# ==========================================
# 4. 图表 B: 升级版排行榜 (Long Bar Chart)
# ==========================================
# 拉长画布高度以容纳 25 个条形
fig, ax = plt.subplots(figsize=(12, 12))

# 创建渐变色
norm = plt.Normalize(df['Probability'].min(), df['Probability'].max())
sm_map = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
palette = [sm_map.to_rgba(x) for x in df['Probability']]

# 绘制水平条形图
bars = sns.barplot(data=df, x='Probability', y='Country', palette=palette, ax=ax)

# 添加数值标签
for i, v in enumerate(df['Probability']):
    ax.text(v + 0.002, i, f"{v:.1%}", va='center', fontsize=10, fontweight='bold', color='#333333')

# 装饰
ax.set_title("Predicted First-Time Olympic Medalists (LA 2028)\nTop 25 Candidates Leaderboard", fontsize=18, weight='bold', pad=20)
ax.set_xlabel("Estimated Probability (ZIP Model)", fontsize=12)
ax.set_ylabel("")

# 添加一条平均线 (可选)
avg_prob = df['Probability'].mean()
ax.axvline(avg_prob, color='gray', linestyle='--', alpha=0.5)
ax.text(avg_prob, 24.5, f' Avg: {avg_prob:.1%}', color='gray', va='center')

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('Chart2_Leaderboard_Top25.png', dpi=300)
print("✅ 图表 2 (Top 25 排行榜) 已保存")

plt.show()