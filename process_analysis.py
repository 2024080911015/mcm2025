import pandas as pd


def generate_total_medal_progress_analysis(pred_file, hist_file, output_file):
    # 1. 加载数据
    df_pred = pd.read_csv(pred_file)
    df_hist = pd.read_csv(hist_file)

    # --- 关键修改点 1: 提取 2024 年【总奖牌数】 ---
    # 假设历史数据中总奖牌列名为 'Total_Medals'
    df_2024 = df_hist[df_hist['Year'] == 2024][['NOC', 'Country_Name', 'Total_Medals']].copy()
    df_2024 = df_2024.rename(columns={'Total_Medals': 'Total_2024'})

    # --- 关键修改点 2: 提取 2028 年【预测总奖牌数】 ---
    # 请检查你的预测文件，确认预测值的列名是 'Predicted_Total_Medals' 还是 'Predicted_Mean' 或其他
    # 这里假设你之前的预测结果列名为 'Predicted_Total_Medals'
    # 如果你的预测文件里列名是 'Predicted_Value' 或 'mean'，请在这里修改
    col_name_in_pred = 'Predicted_Total_Medals'

    # 检查列是否存在，防止报错
    if col_name_in_pred not in df_pred.columns:
        # 尝试自动寻找可能的列名
        candidates = ['Predicted_Total_Medals', 'Predicted_Medals', 'predicted_mean', 'mean']
        for c in candidates:
            if c in df_pred.columns:
                col_name_in_pred = c
                break

    print(f"使用的预测列名为: {col_name_in_pred}")

    df_2028 = df_pred[['NOC', col_name_in_pred]].copy()
    df_2028 = df_2028.rename(columns={col_name_in_pred: 'Predicted_Total_2028'})

    # 4. 合并数据
    merged = pd.merge(df_2024, df_2028, on='NOC', how='inner')

    # 5. 计算变化值 (Delta)
    merged['Delta'] = merged['Predicted_Total_2028'] - merged['Total_2024']

    # 6. 排序 (进步最大的在前面)
    final_df = merged.sort_values('Delta', ascending=False)

    # 7. 整理列顺序
    final_df = final_df[['NOC', 'Country_Name', 'Total_2024', 'Predicted_Total_2028', 'Delta']]

    # 8. 保存为 CSV 文件
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"分析完成！文件已保存为: {output_file}")
    print("\n前5名进步国家 (总奖牌):")
    print(final_df.head())
    print("\n前5名退步国家 (总奖牌):")
    print(final_df.tail())


# --- 执行函数 ---
generate_total_medal_progress_analysis(
    pred_file='prediction_2028_final_with_CI.csv',
    hist_file='modeling_data_maturity.csv',
    output_file='total_medal_progress_regress.csv'
)