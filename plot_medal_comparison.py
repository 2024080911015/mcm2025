import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('total_medal_progress_regress.csv')

# Select 15 representative countries
selected_countries = [
    'CHN',   # China - Large decrease
    'GBR',   # Great Britain - Large decrease
    'FRA',   # France - Large decrease
    'USA',   # United States - Large decrease
    'AUS',   # Australia - Decrease
    'JPN',   # Japan - Decrease
    'KOR',   # South Korea - Decrease
    'ITA',   # Italy - Decrease
    'NED',   # Netherlands - Decrease
    'GER',   # Germany - Small decrease
    'ESP',   # Spain - Small increase
    'POL',   # Poland - Increase
    'IND',   # India - Increase
    'EGY',   # Egypt - Large increase
    'ARG',   # Argentina - Large increase
]

# Filter data
selected_data = df[df['NOC'].isin(selected_countries)].copy()

# Sort by selected order
selected_data['Order'] = selected_data['NOC'].map({country: i for i, country in enumerate(selected_countries)})
selected_data = selected_data.sort_values('Order')

# Set font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))

# Set x-axis positions
x = np.arange(len(selected_data))
width = 0.35

# ========== Upper part: 2024 and 2028 medal counts ==========
bars1 = ax.bar(x - width/2, selected_data['Total_2024'], width, 
               label='2024', color='gold', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, selected_data['Predicted_Total_2028'], width, 
               label='2028', color='royalblue', alpha=0.8, edgecolor='black', linewidth=0.5)

# Display values on upper bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ========== Lower part: Delta values as bars ==========
delta_values = selected_data['Delta'].values
colors = ['red' if d > 0 else 'green' if d < 0 else 'gray' for d in delta_values]

# Draw delta values as negative bars (extending downward)
bars_delta = ax.bar(x, -np.abs(delta_values), width * 2, 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

# Display values on lower bars
for i, (bar, delta) in enumerate(zip(bars_delta, delta_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{delta:+.0f}',
            ha='center', va='top', 
            fontsize=10, fontweight='bold', color=colors[i])

# Set labels
ax.set_xlabel('Country', fontsize=14, fontweight='bold')
ax.set_ylabel('Medal Count / Change', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(selected_data['NOC'], rotation=0, ha='center', fontsize=11)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linewidth=2, linestyle='-')

# Grid
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Legends
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gold', alpha=0.8, edgecolor='black', label='2024'),
    Patch(facecolor='royalblue', alpha=0.8, edgecolor='black', label='2028'),
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Increase'),
    Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Decrease')
]
ax.legend(handles=legend_elements, fontsize=11, loc='upper right', framealpha=0.9)

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('medal_comparison_15_countries.png', dpi=300, bbox_inches='tight')
print("图表已保存为 medal_comparison_15_countries.png")

# 显示图形
plt.show()
