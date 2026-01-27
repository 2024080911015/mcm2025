import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 1. 加载数据
df = pd.read_csv('2028_breakthrough_predictions.csv')

# 2. 准备数据
x = np.arange(len(df))
prob = df['Breakthrough_Prob']
athlete_count = df['Athlete_Count']
event_count = df['Events_Participated_Count']

# 3. 定义颜色
# --- 柱状图颜色 ---
color_low = '#89e5f7'    # 浅蓝
color_mid = '#dec2f7'    # 浅紫
color_high = '#fcaeb3'   # 浅红

bar_colors = []
for p in prob:
    if p > 0.2:
        bar_colors.append(color_high)
    elif p >= 0.12:
        bar_colors.append(color_mid)
    else:
        bar_colors.append(color_low)

# --- 折线图颜色 ---
line_color_athlete = '#5e35b1'  # 深紫色
line_color_event = '#00897b'    # 深青色

# 4. 创建画布
fig, ax1 = plt.subplots(figsize=(14, 8))

# 5. 绘制柱状图 (左侧 Y 轴)
bars = ax1.bar(x, prob, color=bar_colors, alpha=0.8, width=0.6, label='Breakthrough Prob')

ax1.set_xlabel('Sample Index (Data Row)')
ax1.set_ylabel('Breakthrough Probability', fontweight='bold', color='#555555')
ax1.set_ylim(0, max(prob) * 1.2)
ax1.tick_params(axis='y', labelcolor='#555555')

# 6. 创建共享 X 轴的右侧 Y 轴
ax2 = ax1.twinx()

# 7. 绘制折线图 (右侧 Y 轴)
line1, = ax2.plot(x, athlete_count, color=line_color_athlete, marker='o',
                  linewidth=2, markersize=6, label='Athlete Count')

line2, = ax2.plot(x, event_count, color=line_color_event, marker='s',
                  linestyle='--', linewidth=2, markersize=6, label='Events Count')

ax2.set_ylabel('Counts (Athletes & Events)', fontweight='bold', color='#333333')
ax2.tick_params(axis='y', labelcolor='#333333')

# ================= 关键修改开始 =================
# 获取两组数据中的最大值
max_data_value = max(athlete_count.max(), event_count.max())

# 设置右侧 Y 轴的范围 (0 到 最大值的 3 倍)
# 这里的 '3' 是倍数，倍数越大，折线越扁，您可以根据需要改成 2 或 4
ax2.set_ylim(0, max_data_value * 2)
# ================= 关键修改结束 =================

# 8. 标题和网格
plt.title('2D Prediction Analysis: Probability vs. Counts', fontsize=16, pad=20)
ax1.grid(axis='y', linestyle=':', alpha=0.5)

# 9. 自定义组合图例
legend_elements = [
    Patch(facecolor=color_high, label='High Prob (>0.20)'),
    Patch(facecolor=color_mid, label='Mid Prob (0.12-0.20)'),
    Patch(facecolor=color_low, label='Low Prob (<0.12)'),
    Line2D([0], [0], color=line_color_athlete, marker='o', lw=2, label='Athlete Count'),
    Line2D([0], [0], color=line_color_event, marker='s', linestyle='--', lw=2, label='Events Count')
]

ax1.legend(handles=legend_elements, loc='upper left', ncol=2, frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('2d_combo_chart_adjusted_scale.png')
plt.show()