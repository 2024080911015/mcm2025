import pandas as pd
import numpy as np

# ==========================================
# 1. 读取数据
# ==========================================
# 这里的 'data_2024.csv' 指的是您第一步整理好的 2024 年数据
# 如果您的文件名是 data_2028.csv，请将下方文件名改为 'data_2028.csv'
input_file = 'data_2028.csv'
df = pd.read_csv(input_file)


# ==========================================
# 2. 定义预测公式
# ==========================================
def predict_medal(Athletes, Events, Is_Host, Lagged_Share, Sigma):
    # 防止 log(0) 错误，虽然数据通常 > 0
    if Athletes <= 0: Athletes = 1
    if Events <= 0: Events = 1

    score = -0.067612 + 0.014247 * np.log(Athletes) + 0.002205 * np.log(Events) + \
            0.078906 * Is_Host + 0.501635 * Lagged_Share + \
            np.random.normal(loc=0, scale=Sigma)
    return max(0, score)


def predict_gold(Athletes, Events, Is_Host, Lagged_Share, Sigma):
    if Athletes <= 0: Athletes = 1
    if Events <= 0: Events = 1

    score = -0.109768 + \
            0.020919 * np.log(Athletes) + \
            0.002084 * np.log(Events) + \
            0.082456 * Is_Host + \
            0.608438 * Lagged_Share + \
            np.random.normal(loc=0, scale=Sigma)
    return max(0, score)


# ==========================================
# 3. 准备 2028 年预测数据
# ==========================================
df_2028 = df.copy()
df_2028['Year'] = 2028

# ------------------------------------------
# [关键修改] 设置东道主
# ------------------------------------------
# 第一步：将所有国家的 Is_Host 重置为 0 (清除 2024 年法国的标记)
df_2028['Is_Host'] = 0

# 第二步：将美国 (USA) 的 Is_Host 设置为 1
df_2028.loc[df_2028['NOC'] == 'USA', 'Is_Host'] = 1

# 检查一下修改结果 (打印出来确认)
print("Host Check (Should be 1 for USA):")
print(df_2028[df_2028['NOC'] == 'USA'][['NOC', 'Is_Host']])

# ==========================================
# 4. 执行预测
# ==========================================
Sigma = 0  # 设为0以计算期望值

# 获取 2024 年的总奖牌池，用于将预测的“份额”转换回“数量”
total_medals_pool = df['Total_Medals'].sum()
total_gold_pool = df['Gold'].sum()

pred_medal_shares = []
pred_gold_shares = []

for index, row in df_2028.iterrows():
    # 预测总奖牌份额 (Lagged_Share 使用 2024 年的 Medal_Share)
    m_share = predict_medal(row['Athlete_Count'], row['Events_Participated_Count'],
                            row['Is_Host'], row['Medal_Share'], Sigma)
    pred_medal_shares.append(m_share)

    # 预测金牌份额 (Lagged_Share 使用 2024 年的 Gold_Share)
    g_share = predict_gold(row['Athlete_Count'], row['Events_Participated_Count'],
                           row['Is_Host'], row['Gold_Share'], Sigma)
    pred_gold_shares.append(g_share)

df_2028['Predicted_Medal_Share'] = pred_medal_shares
df_2028['Predicted_Gold_Share'] = pred_gold_shares

# 计算具体数量 = 预测份额 * 总池
# round() 四舍五入，astype(int) 转为整数
df_2028['Predicted_Total_Medals'] = (df_2028['Predicted_Medal_Share'] * total_medals_pool).round().astype(int)
df_2028['Predicted_Gold'] = (df_2028['Predicted_Gold_Share'] * total_gold_pool).round().astype(int)

# ==========================================
# 5. 整理输出与导出
# ==========================================
output_columns = ['Year', 'NOC', 'Country_Name', 'Is_Host',
                  'Athlete_Count', 'Events_Participated_Count',
                  'Predicted_Gold', 'Predicted_Total_Medals',
                  'Predicted_Gold_Share', 'Predicted_Medal_Share']

# 按预测金牌数降序排列
df_final = df_2028[output_columns].sort_values(by=['Predicted_Gold', 'Predicted_Total_Medals'], ascending=False)

# 导出 CSV
output_filename = 'prediction_2028_final.csv'
df_final.to_csv(output_filename, index=False)

print(f"\n预测完成！文件已保存为: {output_filename}")
print(df_final.head(10))