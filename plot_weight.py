import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据
df = pd.read_csv('modeling_data_all.csv')

# 2. 定义国家和颜色
countries = ['USA', 'GBR', 'CHN']
colors = {'USA': '#1f77b4', 'GBR': '#d62728', 'CHN': '#2ca02c'} # 蓝、红、绿
labels_map = {'USA': 'USA', 'GBR': 'UK (Great Britain)', 'CHN': 'China'}

# 3. 设置绘图
plt.figure(figsize=(14, 7))

# 4. 绘制背景区域 (权重时期)
x_min = df['Year'].min() - 2
x_max = df['Year'].max() + 2
# Early: < 1945
plt.axvspan(x_min, 1945, color='#a6cee3', alpha=0.3, label='Early Era (Weight ~0.5722)')
# Formation: 1945 - 1992
plt.axvspan(1945, 1992, color='#fdbf6f', alpha=0.3, label='Formation Era (Weight ~0.6822)')
# Maturity: > 1992
plt.axvspan(1992, x_max, color='#fb9a99', alpha=0.3, label='Maturity Era (Weight 1.0000)')

# 5. 循环绘制每个国家的折线
for noc in countries:
    country_data = df[df['NOC'] == noc].sort_values('Year')
    if not country_data.empty:
        plt.plot(country_data['Year'], country_data['Medal_Share'],
                 color=colors[noc], linestyle='-', linewidth=2.5, alpha=0.9, zorder=3,
                 label=labels_map[noc])
        plt.scatter(country_data['Year'], country_data['Medal_Share'],
                    color=colors[noc], s=50, edgecolor='white', zorder=4)

# 6. 添加顶部文字标注
y_max = df[df['NOC'].isin(countries)]['Medal_Share'].max()
#text_pos_y = y_max * 1.05
plt.text(1920, 0.5, 'Weight: 0.5722', fontsize=12, fontweight='bold', color='#1f78b4', ha='center')
plt.text(1968, 0.5, 'Weight: 0.6822', fontsize=12, fontweight='bold', color='#ff7f00', ha='center')
plt.text(2008, 0.5, 'Weight: 1.0000', fontsize=12, fontweight='bold', color='#e31a1c', ha='center')

# 7. 标题、标签和图例
#plt.title('Medal Share Trajectory: USA, UK, and China', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Medal Share', fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(0, y_max * 1.15) # 留出顶部空间

plt.grid(True, linestyle='--', alpha=0.5)
# 设置图例：分两列显示，位置在右上角
plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, fontsize=10, ncol=2)

plt.tight_layout()
plt.show()