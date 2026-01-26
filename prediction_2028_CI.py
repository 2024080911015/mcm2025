import pandas as pd
import numpy as np

# ==========================================
# 1. 读取数据
# ==========================================
input_file = 'data_2028.csv'
df = pd.read_csv(input_file)


# ==========================================
# 2. 定义预测公式 (修改版)
# ==========================================
# 修改点：使用了 np.maximum 替代 max，这样可以一次传进去 10000 个随机数进行并行计算
def predict_medal_sim(Athletes, Events, Is_Host, Lagged_Share, Sigma, n_sims=10000):
    # 防止 log(0)
    if Athletes <= 0: Athletes = 1
    if Events <= 0: Events = 1

    # 计算基础得分（固定部分）
    base_score = -0.067612 + 0.014247 * np.log(Athletes) + 0.002205 * np.log(Events) + \
                 0.078906 * Is_Host + 0.501635 * Lagged_Share

    # 修改点：生成 10000 个随机扰动，而不是 1 个
    random_shocks = np.random.normal(loc=0, scale=Sigma, size=n_sims)

    # 计算 10000 个可能的份额
    scores = base_score + random_shocks
    return np.maximum(0, scores)  # 返回一个包含 10000 个数值的数组


def predict_gold_sim(Athletes, Events, Is_Host, Lagged_Share, Sigma, n_sims=10000):
    if Athletes <= 0: Athletes = 1
    if Events <= 0: Events = 1

    base_score = -0.109768 + 0.020919 * np.log(Athletes) + 0.002084 * np.log(Events) + \
                 0.082456 * Is_Host + 0.608438 * Lagged_Share

    random_shocks = np.random.normal(loc=0, scale=Sigma, size=n_sims)
    scores = base_score + random_shocks
    return np.maximum(0, scores)


# ==========================================
# 3. 准备 2028 年预测数据
# ==========================================
df_2028 = df.copy()
df_2028['Year'] = 2028

# ------------------------------------------
# [关键修改] 设置东道主
# ------------------------------------------
df_2028['Is_Host'] = 0
df_2028.loc[df_2028['NOC'] == 'USA', 'Is_Host'] = 1

print("Host Check (Should be 1 for USA):")
print(df_2028[df_2028['NOC'] == 'USA'][['NOC', 'Is_Host']])

# ==========================================
# 4. 执行预测 (带置信区间)
# ==========================================
# 修改点：不再设为 0，而是使用真实的回归标准差
SIGMA_MEDAL = 0.02729
SIGMA_GOLD = 0.03676
N_SIMS = 10000  # 模拟次数

total_medals_pool = df['Total_Medals'].sum()
total_gold_pool = df['Gold'].sum()

# 用于存放最终结果的列表
results_list = []

print(f"正在进行 {N_SIMS} 次模拟计算...")

for index, row in df_2028.iterrows():
    # 1. 获取 10000 次模拟的份额数组
    medal_shares_sim = predict_medal_sim(row['Athlete_Count'], row['Events_Participated_Count'],
                                         row['Is_Host'], row['Medal_Share'], SIGMA_MEDAL, N_SIMS)

    gold_shares_sim = predict_gold_sim(row['Athlete_Count'], row['Events_Participated_Count'],
                                       row['Is_Host'], row['Gold_Share'], SIGMA_GOLD, N_SIMS)

    # 2. 转换为具体数量 (份额 * 总池)
    medal_counts_sim = medal_shares_sim * total_medals_pool
    gold_counts_sim = gold_shares_sim * total_gold_pool

    # 3. 计算统计指标：均值、下限(2.5%)、上限(97.5%)
    # 金牌
    pred_gold = np.mean(gold_counts_sim)
    pred_gold_share = np.mean(gold_shares_sim)
    gold_lower = np.percentile(gold_counts_sim, 2.5)
    gold_upper = np.percentile(gold_counts_sim, 97.5)

    # 总奖牌
    pred_total = np.mean(medal_counts_sim)
    pred_medal_share = np.mean(medal_shares_sim)
    total_lower = np.percentile(medal_counts_sim, 2.5)
    total_upper = np.percentile(medal_counts_sim, 97.5)

    # 4. 存入结果 (保持与 prediction2028.py 相同的列顺序，并在末尾添加置信区间)
    results_list.append({
        'Year': 2028,
        'NOC': row['NOC'],
        'Country_Name': row['Country_Name'],
        'Is_Host': row['Is_Host'],
        'Athlete_Count': row['Athlete_Count'],
        'Events_Participated_Count': row['Events_Participated_Count'],
        
        # 预测值 (取整)
        'Predicted_Gold': int(round(pred_gold)),
        'Predicted_Total_Medals': int(round(pred_total)),
        
        # 预测份额
        'Predicted_Gold_Share': pred_gold_share,
        'Predicted_Medal_Share': pred_medal_share,

        # 95% 置信区间字符串 (放在最后)
        'Gold_95_CI': f"[{int(round(gold_lower))}, {int(round(gold_upper))}]",
        'Total_Medals_95_CI': f"[{int(round(total_lower))}, {int(round(total_upper))}]"
    })

# ==========================================
# 5. 整理输出与导出
# ==========================================
# 将结果列表转为 DataFrame
df_final = pd.DataFrame(results_list)

# 按预测金牌数降序排列
df_final = df_final.sort_values(by=['Predicted_Gold', 'Predicted_Total_Medals'], ascending=False)

# 导出 CSV
output_filename = 'prediction_2028_final_with_CI.csv'
df_final.to_csv(output_filename, index=False)

print(f"\n预测完成！文件已保存为: {output_filename}")
# 打印前 10 名查看
print(df_final[
          ['NOC', 'Country_Name', 'Predicted_Gold', 'Predicted_Total_Medals', 
           'Predicted_Gold_Share', 'Predicted_Medal_Share', 
           'Gold_95_CI', 'Total_Medals_95_CI']].head(10))