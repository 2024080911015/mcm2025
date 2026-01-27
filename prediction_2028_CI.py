import pandas as pd
import numpy as np

# ==========================================
# 1. 读取数据
# ==========================================
input_file = 'data_2028.csv'
df = pd.read_csv(input_file)


# ==========================================
# 2. 定义预测公式 (单次预测版本)
# ==========================================
# 修改点：每次只预测一个份额值（用于在模拟循环中调用）
def predict_medal_once(Athletes, Events, Is_Host, Lagged_Share, Sigma):
    # 防止 log(0)
    if Athletes <= 0: Athletes = 1
    if Events <= 0: Events = 1

    # 计算基础得分（固定部分）
    base_score = -0.06674 + 0.01406  * np.log(Athletes) + 0.00216  * np.log(Events) + \
            0.07876 * Is_Host + 0.51280  * Lagged_Share


    # 生成单次随机扰动
    random_shock = np.random.normal(loc=0, scale=Sigma)

    # 返回预测份额（未截断，稍后统一处理）
    return base_score + random_shock


def predict_gold_once(Athletes, Events, Is_Host, Lagged_Share, Sigma):
    if Athletes <= 0: Athletes = 1
    if Events <= 0: Events = 1

    base_score =  -0.10724 + \
            0.02033 * np.log(Athletes) + \
            0.00210 * np.log(Events) + \
            0.08233  * Is_Host + \
            0.62531 * Lagged_Share

    random_shock = np.random.normal(loc=0, scale=Sigma)
    return base_score + random_shock


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
# 4. 执行预测 (带置信区间，关键修改：归一化)
# ==========================================
# 关键修改：不再对每个国家独立模拟，而是在每次模拟中同时预测所有国家，然后归一化
SIGMA_MEDAL = 0.02700
SIGMA_GOLD =  0.03600
N_SIMS = 10000  # 模拟次数

total_medals_pool = df['Total_Medals'].sum()
total_gold_pool = df['Gold'].sum()

print(f"正在进行 {N_SIMS} 次模拟计算（带归一化）...")

# 用字典存储每个国家的所有模拟结果
country_gold_sims = {row['NOC']: [] for _, row in df_2028.iterrows()}
country_medal_sims = {row['NOC']: [] for _, row in df_2028.iterrows()}

# 进行 N_SIMS 次模拟
for sim_idx in range(N_SIMS):
    if (sim_idx + 1) % 2000 == 0:
        print(f"  已完成 {sim_idx + 1}/{N_SIMS} 次模拟...")

    # 第一步：对所有国家预测份额（本次模拟）
    gold_shares_this_sim = []
    medal_shares_this_sim = []
    nocs = []

    for index, row in df_2028.iterrows():
        # 预测金牌份额（单次）
        g_share = predict_gold_once(
            row['Athlete_Count'],
            row['Events_Participated_Count'],
            row['Is_Host'],
            row['Gold_Share'],
            SIGMA_GOLD
        )
        gold_shares_this_sim.append(g_share)

        # 预测总奖牌份额（单次）
        m_share = predict_medal_once(
            row['Athlete_Count'],
            row['Events_Participated_Count'],
            row['Is_Host'],
            row['Medal_Share'],
            SIGMA_MEDAL
        )
        medal_shares_this_sim.append(m_share)
        nocs.append(row['NOC'])

    # 第二步：转为数组，截断负值，然后归一化
    gold_shares_this_sim = np.array(gold_shares_this_sim)
    medal_shares_this_sim = np.array(medal_shares_this_sim)

    gold_shares_this_sim = np.maximum(0, gold_shares_this_sim)  # 截断负值
    medal_shares_this_sim = np.maximum(0, medal_shares_this_sim)

    # 关键步骤：归一化，确保份额之和 = 1
    if gold_shares_this_sim.sum() > 0:
        gold_shares_this_sim = gold_shares_this_sim / gold_shares_this_sim.sum()
    if medal_shares_this_sim.sum() > 0:
        medal_shares_this_sim = medal_shares_this_sim / medal_shares_this_sim.sum()

    # 第三步：转换为奖牌数量
    gold_counts_this_sim = gold_shares_this_sim * total_gold_pool
    medal_counts_this_sim = medal_shares_this_sim * total_medals_pool

    # 第四步：记录到每个国家的模拟结果中
    for i, noc in enumerate(nocs):
        country_gold_sims[noc].append(gold_counts_this_sim[i])
        country_medal_sims[noc].append(medal_counts_this_sim[i])

# 用于存放最终结果的列表
results_list = []

print(f"模拟完成，正在计算统计指标...")

# 对每个国家统计模拟结果
for index, row in df_2028.iterrows():
    noc = row['NOC']

    # 获取该国家的所有模拟结果
    gold_sim_array = np.array(country_gold_sims[noc])
    medal_sim_array = np.array(country_medal_sims[noc])

    # 计算统计指标：均值、下限(2.5%)、上限(97.5%)
    # 金牌
    pred_gold = np.mean(gold_sim_array)
    pred_gold_share = pred_gold / total_gold_pool  # 从数量反推份额
    gold_lower = np.percentile(gold_sim_array, 2.5)
    gold_upper = np.percentile(gold_sim_array, 97.5)

    # 总奖牌
    pred_total = np.mean(medal_sim_array)
    pred_medal_share = pred_total / total_medals_pool  # 从数量反推份额
    total_lower = np.percentile(medal_sim_array, 2.5)
    total_upper = np.percentile(medal_sim_array, 97.5)

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
