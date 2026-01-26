import pandas as pd
import numpy as np
import re

# ==========================================
# 1. 基础清理工具：移除乱码、特殊不可见字符
# ==========================================
def aggressive_clean(text):
    if not isinstance(text, str): return text
    return re.sub(r'[^\x20-\x7E]', '', text).strip()

# ==========================================
# 2. 数据加载
# ==========================================
medal_counts = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')
hosts = pd.read_csv("summerOly_hosts.csv", encoding='ISO-8859-1')
programs = pd.read_csv("summerOly_programs.csv", encoding='ISO-8859-1')
athletes = pd.read_csv("summerOly_athletes.csv", encoding='ISO-8859-1')

# ==========================================
# 3. 清洗东道主信息
# ==========================================
hosts.columns = hosts.columns.str.replace('ï»¿', '').str.strip()
hosts = hosts[~hosts['Host'].str.contains('Cancelled', na=False)].copy()
hosts['Host_Country_Clean'] = hosts['Host'].str.split(',').str[-1].apply(aggressive_clean)

# 建立东道主国家名到 NOC 的映射（手动校准核心国家）
host_to_noc_map = {
    'Greece': 'GRE', 'France': 'FRA', 'United States': 'USA', 'United Kingdom': 'GBR',
    'Sweden': 'SWE', 'Belgium': 'BEL', 'Netherlands': 'NED', 'Germany': 'GER',
    'Finland': 'FIN', 'Australia': 'AUS', 'Italy': 'ITA', 'Japan': 'JPN',
    'Mexico': 'MEX', 'West Germany': 'FRG', 'Canada': 'CAN', 'Soviet Union': 'URS',
    'South Korea': 'KOR', 'Spain': 'ESP', 'China': 'CHN', 'Brazil': 'BRA'
}
hosts['Host_NOC'] = hosts['Host_Country_Clean'].map(host_to_noc_map)

# ==========================================
# 4. 计算全球总项目数 (Global Total Events)
# ==========================================
year_cols = [col for col in programs.columns if col.replace('*','').isnumeric()]
programs_long = programs.melt(id_vars=['Sport'], value_vars=year_cols, var_name='Year', value_name='Events')
programs_long['Year'] = programs_long['Year'].str.replace('*', '', regex=False).astype(int)
# 将 '•' 处理为 1，缺失或非数值处理为 0
programs_long['Events_Val'] = programs_long['Events'].apply(
    lambda x: 1 if x == '•' else (float(x) if pd.notna(x) and str(x).replace('.','').isnumeric() else 0)
)
global_events = programs_long.groupby('Year')['Events_Val'].sum().reset_index(name='Global_Total_Events')

# ==========================================
# 5. 聚合国家参赛统计（仅保留数值指标）
# ==========================================
# 剔除 1906 年和混合代表团 ZZX
athletes = athletes[(athletes['Year'] != 1906) & (athletes['NOC'] != 'ZZX')].copy()

# 提取数值特征：人数、大项数、小项数
country_stats = athletes.groupby(['Year', 'NOC']).agg(
    Athlete_Count=('Name', 'nunique'),
    Sports_Participated_Count=('Sport', 'nunique'),
    Events_Participated_Count=('Event', 'nunique')
).reset_index()

# 辅助：获取国家名称映射（用于展示）
noc_to_team = athletes.groupby('NOC')['Team'].agg(lambda x: x.mode()[0]).to_dict()
country_stats['Country_Name'] = country_stats['NOC'].map(noc_to_team)

# ==========================================
# 6. 整合奖牌榜与占比标准化 (Medal Share)
# ==========================================
# 建立奖牌表国家名到代码的自动映射
name_to_noc = {aggressive_clean(k): v for k, v in athletes.groupby('Team')['NOC'].first().to_dict().items()}
# 手动强制修正历史关键更迭
manual_overrides = {
    'United States': 'USA', 'Soviet Union': 'URS', 'Great Britain': 'GBR',
    'West Germany': 'FRG', 'East Germany': 'GDR', 'Russian Empire': 'RUS',
    'Russia': 'RUS', 'China': 'CHN', 'South Korea': 'KOR', 'Chinese Taipei': 'TPE',
    'ROC': 'ROC', 'Netherlands': 'NED', 'Spain': 'ESP', 'Australia': 'AUS', 'Japan': 'JPN'
}
name_to_noc.update(manual_overrides)

medal_counts['NOC_Code'] = medal_counts['NOC'].apply(aggressive_clean).map(name_to_noc)
medal_counts = medal_counts[(medal_counts['Year'] != 1906) & (medal_counts['NOC'] != 'Mixed team')]

# 计算每年全社会的奖牌总池（用于计算 Share）
yearly_pools = medal_counts.groupby('Year').agg(
    Total_Gold_Pool=('Gold', 'sum'),
    Total_Medal_Pool=('Total', 'sum')
).reset_index()

# 合并统计数据与奖牌数据
final_df = pd.merge(country_stats, medal_counts[['Year', 'NOC_Code', 'Gold', 'Total']],
                     left_on=['Year', 'NOC'], right_on=['Year', 'NOC_Code'], how='left')

final_df[['Gold', 'Total']] = final_df[['Gold', 'Total']].fillna(0)
final_df.rename(columns={'Total': 'Total_Medals'}, inplace=True)
final_df.drop(columns=['NOC_Code'], inplace=True)

# 计算核心比例特征 (消除百年扩张偏差)
final_df = pd.merge(final_df, yearly_pools, on='Year', how='left')
final_df['Gold_Share'] = final_df['Gold'] / final_df['Total_Gold_Pool']
final_df['Medal_Share'] = final_df['Total_Medals'] / final_df['Total_Medal_Pool']

# ==========================================
# 7. 时代分类与东道主判定
# ==========================================
def get_era(year):
    if year <= 1928: return 'Early'
    if year <= 1988: return 'Formation'
    return 'Maturity'
final_df['Era'] = final_df['Year'].apply(get_era)

final_df = pd.merge(final_df, hosts[['Year', 'Host_NOC']], on='Year', how='left')
final_df['Is_Host'] = (final_df['NOC'] == final_df['Host_NOC']).astype(int)
final_df.drop(columns=['Host_NOC'], inplace=True)

# 合并全球事件背景
final_df = pd.merge(final_df, global_events, on='Year', how='left')

# ==========================================
# 8. 输出最终纯净版数据
# ==========================================
final_columns = [
    'Year', 'Era', 'NOC', 'Country_Name', 'Athlete_Count',
    'Sports_Participated_Count', 'Events_Participated_Count',
    'Global_Total_Events', 'Is_Host', 'Gold', 'Total_Medals',
    'Gold_Share', 'Medal_Share'
]
final_df = final_df[final_columns].sort_values(['Year', 'Total_Medals'], ascending=[True, False])

# 保存文件
final_df.to_csv('olympic_modeling_numeric.csv', index=False)
# 额外保存一份 1992 年以后的数据供主模型使用
final_df[final_df['Era'] == 'Maturity'].to_csv('olympic_modern_maturity.csv', index=False)

print("数据清洗完成！已生成纯数值化的建模数据集。")
print(f"全量数据: {len(final_df)} 条")