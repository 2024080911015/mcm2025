import pandas as pd
import numpy as np
import re

# ==========================================
# 1. 基础清理工具：移除乱码、特殊不可见字符
# ==========================================
def aggressive_clean(text):
    if not isinstance(text, str): return text
    # 仅保留标准打印字符，去除 Â 等乱码
    return re.sub(r'[^\x20-\x7E]', '', text).strip()

# ==========================================
# 2. 数据加载
# ==========================================
# 请确保这四个原始文件在你的当前工作目录下
medal_counts = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')
hosts = pd.read_csv("summerOly_hosts.csv", encoding='ISO-8859-1')
programs = pd.read_csv("summerOly_programs.csv", encoding='ISO-8859-1')
athletes = pd.read_csv("summerOly_athletes.csv", encoding='ISO-8859-1')

# ==========================================
# 3. 清洗东道主数据
# ==========================================
hosts.columns = hosts.columns.str.replace('ï»¿', '').str.strip()
# 剔除取消的年份
hosts = hosts[~hosts['Host'].str.contains('Cancelled', na=False)].copy()
hosts['Host_Country_Clean'] = hosts['Host'].str.split(',').str[-1].apply(aggressive_clean)

# 核心东道主国家/地区到 NOC 的映射
host_to_noc_map = {
    'Greece': 'GRE', 'France': 'FRA', 'United States': 'USA', 'United Kingdom': 'GBR',
    'Sweden': 'SWE', 'Belgium': 'BEL', 'Netherlands': 'NED', 'Germany': 'GER',
    'Finland': 'FIN', 'Australia': 'AUS', 'Italy': 'ITA', 'Japan': 'JPN',
    'Mexico': 'MEX', 'West Germany': 'FRG', 'Canada': 'CAN', 'Soviet Union': 'URS',
    'South Korea': 'KOR', 'Spain': 'ESP', 'China': 'CHN', 'Brazil': 'BRA'
}
hosts['Host_NOC'] = hosts['Host_Country_Clean'].map(host_to_noc_map)

# ==========================================
# 4. 计算全球总项目数 (Global Context)
# ==========================================
year_cols = [col for col in programs.columns if col.replace('*','').isnumeric()]
programs_long = programs.melt(id_vars=['Sport'], value_vars=year_cols, var_name='Year', value_name='Events')
programs_long['Year'] = programs_long['Year'].str.replace('*', '', regex=False).astype(int)
# 处理符号并求和
programs_long['Events_Val'] = programs_long['Events'].apply(
    lambda x: 1 if x == '•' else (float(x) if pd.notna(x) and str(x).replace('.','').replace('-','').isnumeric() else 0)
)
global_events = programs_long.groupby('Year')['Events_Val'].sum().reset_index(name='Global_Total_Events')

# ==========================================
# 5. 处理苏联/俄罗斯及国家统计 (纯数值)
# ==========================================
# 1. 过滤噪音
athletes = athletes[(athletes['Year'] != 1906) & (athletes['NOC'] != 'ZZX')].copy()

# 2. 苏联与俄罗斯的特殊逻辑 (逻辑 A: 1992年后的继承者统一为 RUS)
# 这样可以保证 AIN(2024), ROC(2020), EUN(1992) 的数据都归在 RUS 的历史序列中
soviet_successors = {'EUN': 'RUS', 'ROC': 'RUS', 'AIN': 'RUS'}
athletes['NOC'] = athletes.apply(lambda x: soviet_successors.get(x['NOC'], x['NOC']) if x['Year'] >= 1992 else x['NOC'], axis=1)

# 3. 聚合数值特征
country_stats = athletes.groupby(['Year', 'NOC']).agg(
    Athlete_Count=('Name', 'nunique'),           # 参赛人数
    Sports_Participated_Count=('Sport', 'nunique'), # 参加大项数
    Events_Participated_Count=('Event', 'nunique')  # 参加小项数
).reset_index()

# 国家名称参考映射
noc_to_name = athletes.groupby('NOC')['Team'].agg(lambda x: x.mode()[0]).to_dict()
country_stats['Country_Name'] = country_stats['NOC'].map(noc_to_name)

# ==========================================
# 6. 整合奖牌榜与份额计算 (Share)
# ==========================================
# 处理奖牌表的映射匹配
team_to_noc = {aggressive_clean(k): v for k, v in athletes.groupby('Team')['NOC'].first().to_dict().items()}
manual_overrides = {
    'United States': 'USA', 'Soviet Union': 'URS', 'Great Britain': 'GBR',
    'West Germany': 'FRG', 'East Germany': 'GDR', 'Russian Empire': 'RUS',
    'Russia': 'RUS', 'China': 'CHN', 'South Korea': 'KOR', 'Chinese Taipei': 'TPE',
    'ROC': 'RUS', 'Unified Team': 'RUS', 'Netherlands': 'NED'
}
team_to_noc.update(manual_overrides)

medal_counts['NOC_Code'] = medal_counts['NOC'].apply(aggressive_clean).map(team_to_noc)
medal_counts = medal_counts[(medal_counts['Year'] != 1906) & (medal_counts['NOC'] != 'Mixed team')].copy()

# 计算每年的奖牌池
yearly_pools = medal_counts.groupby('Year').agg(
    Total_Gold_Pool=('Gold', 'sum'),
    Total_Medal_Pool=('Total', 'sum')
).reset_index()

# 合并奖牌
final_df = pd.merge(country_stats, medal_counts[['Year', 'NOC_Code', 'Gold', 'Total']],
                     left_on=['Year', 'NOC'], right_on=['Year', 'NOC_Code'], how='left')
final_df[['Gold', 'Total']] = final_df[['Gold', 'Total']].fillna(0)
final_df.rename(columns={'Total': 'Total_Medals'}, inplace=True)

# 计算 Share 指标
final_df = pd.merge(final_df, yearly_pools, on='Year', how='left')
final_df['Gold_Share'] = final_df['Gold'] / final_df['Total_Gold_Pool']
final_df['Medal_Share'] = final_df['Total_Medals'] / final_df['Total_Medal_Pool']

# ==========================================
# 7. 时代分类与东道主判定
# ==========================================
def get_era(year):
    if year <= 1945: return 'Early'      # 早期
    if year <= 1991: return 'Formation'  # 形成期 (冷战)
    return 'Maturity'                   # 成熟期

final_df['Era'] = final_df['Year'].apply(get_era)

# 东道主
final_df = pd.merge(final_df, hosts[['Year', 'Host_NOC']], on='Year', how='left')
final_df['Is_Host'] = (final_df['NOC'] == final_df['Host_NOC']).astype(int)

# 全球事件总数
final_df = pd.merge(final_df, global_events, on='Year', how='left')

# ==========================================
# 8. 最终导出
# ==========================================
final_cols = [
    'Year', 'Era', 'NOC', 'Country_Name', 'Is_Host', 'Athlete_Count',
    'Sports_Participated_Count', 'Events_Participated_Count', 'Global_Total_Events',
    'Gold', 'Total_Medals', 'Gold_Share', 'Medal_Share'
]
final_df = final_df[final_cols].sort_values(['Year', 'Total_Medals'], ascending=[True, False])

# 导出四个阶段
final_df.to_csv("modeling_data_all.csv", index=False)
final_df[final_df['Era'] == 'Early'].to_csv("modeling_data_early.csv", index=False)
final_df[final_df['Era'] == 'Formation'].to_csv("modeling_data_formation.csv", index=False)
final_df[final_df['Era'] == 'Maturity'].to_csv("modeling_data_maturity.csv", index=False)

print("✅ 数据处理与分段导出完成！")