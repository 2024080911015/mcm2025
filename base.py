import pandas as pd
import numpy as np
import re

# ==========================================
# 1. 数据加载与基础清洗
# ==========================================
# 确保文件在当前目录下
medal_counts = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')
programs = pd.read_csv("summerOly_programs.csv", encoding='ISO-8859-1')
athletes = pd.read_csv("summerOly_athletes.csv", encoding='ISO-8859-1')

# 剔除噪音数据 (1906年届间奥运会, 混合代表团)
medal_counts = medal_counts[(medal_counts['Year'] != 1906) & (medal_counts['NOC'] != 'Mixed team')].copy()
athletes = athletes[(athletes['Year'] != 1906) & (athletes['NOC'] != 'ZZX')].copy()

# ==========================================
# 2. 计算三个客观指标 (按年统计)
# ==========================================

# 指标 1: NOC Count (参赛国家数 - 规模维度)
# 使用 athletes 表统计更为准确，代表实际派出运动员的国家数
yearly_noc_count = athletes.groupby('Year')['NOC'].nunique().reset_index(name='NOC_Count')

# 指标 2: Total Events (总设项数 - 结构维度)
# 清洗 programs 表，转为长表格式
year_cols = [col for col in programs.columns if col.replace('*', '').isnumeric()]
programs_long = programs.melt(id_vars=['Sport'], value_vars=year_cols, var_name='Year_Raw', value_name='Events')
programs_long['Year'] = programs_long['Year_Raw'].str.replace('*', '', regex=False).astype(int)
# 将 '•' 转换为 1，其余转为数值
programs_long['Event_Val'] = programs_long['Events'].apply(
    lambda x: 1 if x == '•' else (float(x) if pd.notna(x) and str(x).replace('.', '').isnumeric() else 0))
# 按年求和
global_events = programs_long.groupby('Year')['Event_Val'].sum().reset_index(name='Total_Events')


# 指标 3: Concentration (Top 5 奖牌占比 - 竞争维度)
def get_concentration(group):
    total_medals = group['Total'].sum()
    if total_medals == 0: return 0
    # 计算前5名国家的奖牌总和 / 当年总奖牌
    top5_sum = group.nlargest(5, 'Total')['Total'].sum()
    return top5_sum / total_medals


yearly_concentration = medal_counts.groupby('Year').apply(get_concentration).reset_index(name='Concentration')

# 合并所有指标
yearly_stats = yearly_noc_count.merge(global_events, on='Year', how='inner').merge(yearly_concentration, on='Year',
                                                                                   how='inner')


# ==========================================
# 3. 划分时代并计算均值 (X_p,k)
# ==========================================
def get_era(year):
    if year <= 1945: return 'Early'  # 早期
    if year <= 1991: return 'Formation'  # 形成期
    return 'Maturity'  # 成熟期 (现代基准)


yearly_stats['Era'] = yearly_stats['Year'].apply(get_era)

# 计算各时代的指标均值向量
era_means = yearly_stats.groupby('Era')[['NOC_Count', 'Total_Events', 'Concentration']].mean()

# ==========================================
# 4. 计算制度偏离度 (Dp) 与 最终权重 (Weight)
# ==========================================
# 设定基准向量: 现代成熟期
X_modern = era_means.loc['Maturity']
K = 3  # 指标数量


def calculate_metrics(row):
    # 计算相对偏离度: |X_p - X_modern| / X_modern
    # 这一步量化了“这个时期与现代奥运有多不像”
    diffs = np.abs((row - X_modern) / X_modern)

    # Dp = 平均偏离度
    Dp = diffs.sum() / K

    # 映射为权重: wp = 1 / (1 + Dp)
    Weight = 1 / (1 + Dp)

    return pd.Series({'Dp': Dp, 'Weight': Weight})


# 应用计算
results = era_means.apply(calculate_metrics, axis=1)
final_table = pd.concat([era_means, results], axis=1)

# 按时间顺序重排显示
final_table = final_table.reindex(['Early', 'Formation', 'Maturity'])

# ==========================================
# 5. 输出结果
# ==========================================
print("--- 基于制度可比性的权重计算结果 ---")
print(final_table)

# 保存文件供后续建模调用
final_table.to_csv('era_weights.csv')
print("\n已保存为 era_weights.csv")