import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 0. 路径设置 (修改点)
# ==========================================
# 只要脚本和CSV在同一目录，直接使用文件名即可
# 如果您在 PyCharm/VSCode 中运行遇到 "File Not Found"，
# 请确保终端(Terminal)的当前目录是脚本所在目录
medal_counts_file = 'summerOly_medal_counts.csv'
hosts_file = 'summerOly_hosts.csv'
programs_file = 'summerOly_programs.csv'
athletes_file = 'summerOly_athletes.csv'


# ==========================================
# 1. 基础清理工具
# ==========================================
def aggressive_clean(text):
    if not isinstance(text, str): return text
    # 移除不可见字符，保留基础ASCII
    return re.sub(r'[^\x20-\x7E]', '', text).strip()


try:
    # ==========================================
    # 2. 数据加载
    # ==========================================
    print("正在加载数据...")
    # 检查文件是否存在
    for f in [medal_counts_file, hosts_file, programs_file, athletes_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件: {f}，请确保它和脚本在同一个文件夹内！")

    medal_counts = pd.read_csv(medal_counts_file, encoding='ISO-8859-1')
    hosts = pd.read_csv(hosts_file, encoding='ISO-8859-1')
    programs = pd.read_csv(programs_file, encoding='ISO-8859-1')
    athletes = pd.read_csv(athletes_file, encoding='ISO-8859-1')
    print("数据加载成功！")

    # ==========================================
    # 3. 清洗东道主信息
    # ==========================================
    hosts.columns = hosts.columns.str.replace('ï»¿', '').str.strip()
    hosts = hosts[~hosts['Host'].str.contains('Cancelled', na=False)].copy()
    hosts['Host_Country_Clean'] = hosts['Host'].str.split(',').str[-1].apply(aggressive_clean)

    host_to_noc_map = {
        'Greece': 'GRE', 'France': 'FRA', 'United States': 'USA', 'United Kingdom': 'GBR',
        'Sweden': 'SWE', 'Belgium': 'BEL', 'Netherlands': 'NED', 'Germany': 'GER',
        'Finland': 'FIN', 'Australia': 'AUS', 'Italy': 'ITA', 'Japan': 'JPN',
        'Mexico': 'MEX', 'West Germany': 'FRG', 'Canada': 'CAN', 'Soviet Union': 'URS',
        'South Korea': 'KOR', 'Spain': 'ESP', 'China': 'CHN', 'Brazil': 'BRA'
    }
    hosts['Host_NOC'] = hosts['Host_Country_Clean'].map(host_to_noc_map)

    # ==========================================
    # 4. 计算全球总项目数
    # ==========================================
    year_cols = [col for col in programs.columns if col.replace('*', '').isnumeric()]
    programs_long = programs.melt(id_vars=['Sport'], value_vars=year_cols, var_name='Year', value_name='Events')
    programs_long['Year'] = programs_long['Year'].str.replace('*', '', regex=False).astype(int)
    programs_long['Events_Val'] = programs_long['Events'].apply(
        lambda x: 1 if x == '•' else (float(x) if pd.notna(x) and str(x).replace('.', '').isnumeric() else 0)
    )
    global_events = programs_long.groupby('Year')['Events_Val'].sum().reset_index(name='Global_Total_Events')

    # ==========================================
    # 5. 聚合国家参赛统计
    # ==========================================
    athletes = athletes[(athletes['Year'] != 1906) & (athletes['NOC'] != 'ZZX')].copy()
    country_stats = athletes.groupby(['Year', 'NOC']).agg(
        Athlete_Count=('Name', 'nunique'),
        Sports_Participated_Count=('Sport', 'nunique'),
        Events_Participated_Count=('Event', 'nunique')
    ).reset_index()

    noc_to_team = athletes.groupby('NOC')['Team'].agg(lambda x: x.mode()[0]).to_dict()
    country_stats['Country_Name'] = country_stats['NOC'].map(noc_to_team)

    # ==========================================
    # 6. 整合奖牌榜
    # ==========================================
    name_to_noc = {aggressive_clean(k): v for k, v in athletes.groupby('Team')['NOC'].first().to_dict().items()}
    manual_overrides = {
        'United States': 'USA', 'Soviet Union': 'URS', 'Great Britain': 'GBR',
        'West Germany': 'FRG', 'East Germany': 'GDR', 'Russian Empire': 'RUS',
        'Russia': 'RUS', 'China': 'CHN', 'South Korea': 'KOR', 'Chinese Taipei': 'TPE',
        'ROC': 'ROC', 'Netherlands': 'NED', 'Spain': 'ESP', 'Australia': 'AUS', 'Japan': 'JPN'
    }
    name_to_noc.update(manual_overrides)

    medal_counts['NOC_Code'] = medal_counts['NOC'].apply(aggressive_clean).map(name_to_noc)
    medal_counts = medal_counts[(medal_counts['Year'] != 1906) & (medal_counts['NOC'] != 'Mixed team')]

    yearly_pools = medal_counts.groupby('Year').agg(
        Total_Gold_Pool=('Gold', 'sum'),
        Total_Medal_Pool=('Total', 'sum')
    ).reset_index()

    final_df = pd.merge(country_stats, medal_counts[['Year', 'NOC_Code', 'Gold', 'Total']],
                        left_on=['Year', 'NOC'], right_on=['Year', 'NOC_Code'], how='left')

    final_df[['Gold', 'Total']] = final_df[['Gold', 'Total']].fillna(0)
    final_df.rename(columns={'Total': 'Total_Medals'}, inplace=True)
    final_df.drop(columns=['NOC_Code'], inplace=True)

    final_df = pd.merge(final_df, yearly_pools, on='Year', how='left')
    final_df['Gold_Share'] = final_df['Gold'] / final_df['Total_Gold_Pool']
    final_df['Medal_Share'] = final_df['Total_Medals'] / final_df['Total_Medal_Pool']


    # ==========================================
    # 7. 时代分类与输出
    # ==========================================
    def get_era(year):
        if year <= 1928: return 'Early'
        if year <= 1988: return 'Formation'
        return 'Maturity'


    final_df['Era'] = final_df['Year'].apply(get_era)

    final_df = pd.merge(final_df, hosts[['Year', 'Host_NOC']], on='Year', how='left')
    final_df['Is_Host'] = (final_df['NOC'] == final_df['Host_NOC']).astype(int)
    final_df.drop(columns=['Host_NOC'], inplace=True)

    final_df = pd.merge(final_df, global_events, on='Year', how='left')

    final_columns = [
        'Year', 'Era', 'NOC', 'Country_Name', 'Athlete_Count',
        'Sports_Participated_Count', 'Events_Participated_Count',
        'Global_Total_Events', 'Is_Host', 'Gold', 'Total_Medals',
        'Gold_Share', 'Medal_Share'
    ]
    final_df = final_df[final_columns].sort_values(['Year', 'Total_Medals'], ascending=[True, False])

    # 保存文件到当前目录
    output_filename = 'olympic_modeling_numeric.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"处理完成！文件已保存为: {output_filename}")
    print(final_df.head().to_string())

except Exception as e:
    print(f"运行出错: {e}")
    # 如果是文件找不到，给个提示
    if "No such file" in str(e) or "not found" in str(e).lower():
        print("\n>>> 提示：请检查您的终端(Terminal)当前路径是否是代码所在的文件夹。")
        print(f">>> 当前工作目录是: {os.getcwd()}")