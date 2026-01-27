import pandas as pd
import numpy as np


def generate_2028_break_zero_analysis(history_file, prediction_file, output_file):
    # 1. 加载数据
    print("正在加载数据...")
    # 历史数据 (用于确定谁从未获奖，以及估算参赛项目数)
    df_history = pd.read_csv(history_file)
    # 2028 预测数据 (提供运动员人数 Athlete_Count)
    df_2028 = pd.read_csv(prediction_file)

    # 2. 数据清洗 (统一去除空格)
    if 'Country_Name' in df_history.columns:
        df_history['Country_Name'] = df_history['Country_Name'].astype(str).str.strip()
    if 'Country_Name' in df_2028.columns:
        df_2028['Country_Name'] = df_2028['Country_Name'].astype(str).str.strip()

    # 3. 锁定“从未获奖国家”名单
    # 计算历史总奖牌
    country_totals = df_history.groupby('Country_Name')['Total_Medals'].sum()
    candidates = country_totals[country_totals == 0].index.tolist()

    # 排除黑名单 (历史错误数据、非国家实体、已消失政权)
    exclude_list = [
        'West Indies Federation', 'United Arab Republic',
        'Saar', 'Malaya', 'North Borneo', 'South Yemen', 'North Yemen',
        'South Vietnam', 'Newfoundland', 'Rhodesia',
        'Czechoslovakia', 'East Germany', 'Serbia and Montenegro', 'Soviet Union',
        'Refugee Olympic Athletes', 'Individual Olympic Athletes', 'Unknown', 'Mixed team'
    ]
    valid_countries = [c for c in candidates if c not in exclude_list]
    print(f"历史数据中确认了 {len(valid_countries)} 个从未获奖的国家。")

    # 4. 准备 2028 年数据
    # 从 2028 预测表中筛选出这些国家
    df_2028_filtered = df_2028[df_2028['Country_Name'].isin(valid_countries)].copy()

    # 5. 补充缺失变量
    # (1) Sports_Participated_Count: 2028 预测表里可能没有这个，我们用该国最近一次参赛的数据作为估计
    # 按年份排序，取每个国家最后一条记录的 Sports_Participated_Count
    last_sports_count = df_history.sort_values('Year').groupby('Country_Name')['Sports_Participated_Count'].last()

    # 映射回 2028 表
    df_2028_filtered['Sports_Participated_Count'] = df_2028_filtered['Country_Name'].map(last_sports_count)

    # 如果仍有缺失 (可能是新国家或匹配不上)，用 Events_Participated_Count 估算或填默认值
    # 这里做一个简单的填充：如果没有历史数据，假设参加 1 个大项
    df_2028_filtered['Sports_Participated_Count'] = df_2028_filtered['Sports_Participated_Count'].fillna(1)

    # (2) Lagged_Total_Medals: 因为从未获奖，所以始终为 0
    df_2028_filtered['Lagged_Total_Medals'] = 0

    # (3) Is_Host: 2028 是美国，这些国家都不是东道主，设为 0
    df_2028_filtered['Is_Host'] = 0

    # 6. 计算破零概率 (应用您的公式)
    # 使用 numpy 向量化计算
    team_size = df_2028_filtered['Athlete_Count']
    sports_count = df_2028_filtered['Sports_Participated_Count']
    lagged_Medals = df_2028_filtered['Lagged_Total_Medals']
    Is_Host = df_2028_filtered['Is_Host']

    # 公式
    df_2028_filtered['score'] = 2.9686 - 0.4066 * team_size + 0.4407 * sports_count
    df_2028_filtered['pi'] = 1 / (1 + np.exp(-df_2028_filtered['score']))
    df_2028_filtered['lambda1'] = np.exp(1.6741 + 0.0316 * lagged_Medals + 1.4404 * Is_Host)
    # p = (1 - pi) * (1 - exp(-lambda1))
    df_2028_filtered['prob_break_zero'] = (1 - df_2028_filtered['pi']) * (1 - np.exp(-df_2028_filtered['lambda1']))

    # 7. 整理输出列
    output_cols = [
        'Year', 'Country_Name', 'NOC',
        'Athlete_Count', 'Sports_Participated_Count',
        'score', 'pi', 'lambda1', 'prob_break_zero'
    ]
    result_df = df_2028_filtered[output_cols].sort_values('prob_break_zero', ascending=False)

    # 8. 保存文件
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("-" * 50)
    print(f"分析完成！文件已保存为: {output_file}")
    print(f"共包含 {len(result_df)} 个国家。")
    print("\n2028年最有希望破零的前5名：")
    print(result_df[['Country_Name', 'Athlete_Count', 'prob_break_zero']].head().to_string(index=False))


# --- 运行配置 ---
# 请确保当前目录下有这两个文件
generate_2028_break_zero_analysis(
    history_file='modeling_data_all.csv',  # 用于筛选名单和获取历史参赛项目数
    prediction_file='prediction_2028_final_with_CI.csv',  # 用于获取2028运动员预测数
    output_file='never_medaled_2028_prob.csv'  # 输出文件名
)