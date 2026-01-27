import pandas as pd


def generate_gold_progress_analysis(pred_file, hist_file, output_file):
    # 1. 加载数据
    # prediction_2028_final_with_CI.csv: 包含 2028 年的预测数据
    df_pred = pd.read_csv(pred_file)
    # modeling_data_maturity.csv: 包含历史数据，我们需要从中提取 2024 年的实际结果
    df_hist = pd.read_csv(hist_file)

    # 2. 提取 2024 年实际金牌数
    # 筛选 Year 为 2024 的数据
    df_2024 = df_hist[df_hist['Year'] == 2024][['NOC', 'Country_Name', 'Gold']].copy()
    # 重命名列以区分年份
    df_2024 = df_2024.rename(columns={'Gold': 'Gold_2024'})

    # 3. 提取 2028 年预测金牌数
    df_2028 = df_pred[['NOC', 'Predicted_Gold']].copy()
    # 重命名列
    df_2028 = df_2028.rename(columns={'Predicted_Gold': 'Predicted_Gold_2028'})

    # 4. 合并数据
    # 使用 inner join 确保只分析在两年都存在数据的国家
    merged = pd.merge(df_2024, df_2028, on='NOC', how='inner')

    # 5. 计算变化值 (Delta)
    # Delta = 2028预测值 - 2024实际值
    merged['Delta'] = merged['Predicted_Gold_2028'] - merged['Gold_2024']

    # 6. 排序
    # 按照变化值降序排列：进步最大的在最前面，退步最大的在最后面
    final_df = merged.sort_values('Delta', ascending=False)

    # 7. 整理列顺序
    # 只保留关键列，使 CSV 更整洁
    final_df = final_df[['NOC', 'Country_Name', 'Gold_2024', 'Predicted_Gold_2028', 'Delta']]

    # 8. 保存为 CSV 文件
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 以防中文乱码

    print(f"分析完成！文件已保存为: {output_file}")
    print("前5名进步国家 (金牌):")
    print(final_df.head())
    print("\n前5名退步国家 (金牌):")
    print(final_df.tail())


# --- 执行函数 ---
# 请确保当前目录下有这两个输入文件
generate_gold_progress_analysis(
    pred_file='prediction_2028_final_with_CI.csv',
    hist_file='modeling_data_maturity.csv',
    output_file='gold_medal_progress_regress.csv'
)