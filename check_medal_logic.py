import pandas as pd
from pathlib import Path


def generate_never_medaled_report(medal_file, participation_file, output_file):
    # 1. 读取数据
    print(f"读取奖牌数据: {medal_file}")
    df_medals = pd.read_csv(medal_file)

    print(f"读取参赛数据: {participation_file}")
    df_partic = pd.read_csv(participation_file)

    # 2. 数据清洗：统一去除名称首尾空格，防止因空格导致匹配失败
    # 注意：summerOly_medal_counts.csv 中的 'NOC' 列实际上是国家名称
    df_medals['NOC'] = df_medals['NOC'].astype(str).str.strip()
    df_partic['Country_Name'] = df_partic['Country_Name'].astype(str).str.strip()

    # 3. 建立“获奖国家”集合 (Source of Truth)
    winners = set(df_medals['NOC'].unique())

    # [关键步骤] 处理名称不一致的问题
    # 例如：奖牌表中可能是 "Virgin Islands"，但参赛表中是 "United States Virgin Islands"
    # 我们需要手动添加映射，防止这些国家被错误地归类为“未获奖”
    if "Virgin Islands" in winners:
        winners.add("United States Virgin Islands")
    # 如果有其他已知别名，也可以在这里添加

    print(f"历史上共有 {len(winners)} 个国家/地区获得过奖牌。")

    # 4. 聚合参赛数据 (获取每个国家的完整参赛历史)
    # 我们需要保留 NOC 代码、国家名，并计算总运动员数、总项目数，列出所有年份
    noc_stats = df_partic.groupby(['NOC', 'Country_Name'], as_index=False).agg({
        'Year': lambda x: sorted(list(set(x))),  # 列出所有参赛年份（去重排序）
        'Athlete_Count': 'sum',  # 累计派出运动员人数
        'Events_Participated_Count': 'sum',  # 累计参加项目数
        'Is_Host': 'sum'  # 举办次数
    })

    # 5. 核心逻辑：筛选从未获奖的国家
    # 条件：国家名 不在 winners 集合中
    never_medaled_df = noc_stats[~noc_stats['Country_Name'].isin(winners)].copy()

    # 6. 整理结果
    # 添加“参赛届数”列，方便排序
    never_medaled_df['Participation_Count'] = never_medaled_df['Year'].apply(len)

    # 按照 参赛届数(降序) -> 派出运动员数(降序) 排序
    # 这样排在前面的是最“遗憾”的国家（参赛多、人多、但无牌）
    never_medaled_df = never_medaled_df.sort_values(
        ['Participation_Count', 'Athlete_Count'],
        ascending=[False, False]
    )

    # 7. 导出 CSV
    never_medaled_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"分析完成！从未拿过奖牌的国家共有: {len(never_medaled_df)} 个")
    print(f"结果已保存至: {output_file}")
    print("-" * 30)
    print("前 5 名（按参赛次数排序）：")
    print(never_medaled_df[['Country_Name', 'Participation_Count', 'Athlete_Count', 'Year']].head().to_string(
        index=False))


# --- 执行配置 ---
# 请根据您本地的实际文件名修改路径
# participation_file 建议使用您之前的 modeling_data_maturity.csv 或 modeling_data_all.csv
generate_never_medaled_report(
    medal_file='summerOly_medal_counts.csv',  # 您的新文件（来源）
    participation_file='modeling_data_maturity.csv',  # 您的参赛信息文件
    output_file='never_medaled_countries_verified.csv'
)