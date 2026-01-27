import pandas as pd
import numpy as np
from scipy.stats import norm


# ==========================================
# 1. 核心预测函数 (基于 prediction_2028_CI.py)
# ==========================================
def predict_mit_tobit(athletes, events, is_host, lag_share_total, lag_share_gold):
    """
    使用 Tobit 模型系数计算预测份额
    """
    # Total Medal Share Formula
    mu_total = -0.06674 + \
               0.01406 * (np.log(athletes) if athletes > 0 else 0) + \
               0.00216 * (np.log(events) if events > 0 else 0) + \
               0.07876 * is_host + \
               0.51280 * lag_share_total
    sigma_total = 0.02700

    # Gold Share Formula（你目前没输出金牌，但这里先不动）
    mu_gold = -0.10724 + \
              0.02033 * (np.log(athletes) if athletes > 0 else 0) + \
              0.00210 * (np.log(events) if events > 0 else 0) + \
              0.08233 * is_host + \
              0.62531 * lag_share_gold
    sigma_gold = 0.03600

    def tobit_exp(mu, sigma):
        z = mu / sigma
        return mu * norm.cdf(z) + sigma * norm.pdf(z)

    return tobit_exp(mu_total, sigma_total), tobit_exp(mu_gold, sigma_gold)


# ==========================================
# 2. 数据准备与计算
# ==========================================
def run_analysis_and_save():
    try:
        df = pd.read_csv('summerOly_athletes.csv')
    except FileNotFoundError:
        print("错误: 找不到 summerOly_athletes.csv 文件，请确认文件路径。")
        return

    def get_stats(df, sport, year, noc, sex_filter=None):
        """
        返回：
        ath_count, evt_count, share_total, share_gold, my_medal_cnt, total_medal_cnt
        """
        # Cycling Track 特殊过滤
        if sport == 'Cycling Track':
            temp = df[(df['Year'] == year) & (df['Sport'] == 'Cycling')]
            exclude = ['Mountainbike', 'Road Race', 'Individual Time Trial', 'BMX']
            pat = '|'.join(exclude)
            temp = temp[~temp['Event'].str.contains(pat, case=False, regex=True)]
        elif sport == 'Volleyball Women':
            temp = df[(df['Year'] == year) & (df['Sport'] == 'Volleyball') & (df['Event'].str.contains("Women"))]
        else:
            temp = df[(df['Year'] == year) & (df['Sport'] == sport)]
            if sex_filter:
                temp = temp[temp['Sex'] == sex_filter]

        # 奖牌统计（按国家口径：Event+Medal+NOC 去重）
        medaled = temp[temp['Medal'].isin(['Gold', 'Silver', 'Bronze'])]
        all_medals = medaled.drop_duplicates(subset=['Event', 'Medal', 'NOC'])
        my_medals = all_medals[all_medals['NOC'] == noc]

        total_medal_cnt = len(all_medals)
        my_medal_cnt = len(my_medals)

        share_total = my_medal_cnt / total_medal_cnt if total_medal_cnt > 0 else 0

        # 金牌份额（你目前没输出，但 lag_share_gold 还在用）
        all_golds = all_medals[all_medals['Medal'] == 'Gold']
        my_golds = my_medals[my_medals['Medal'] == 'Gold']
        share_gold = len(my_golds) / len(all_golds) if len(all_golds) > 0 else 0

        # 特征（TeamSize / Events）
        my_team = temp[temp['NOC'] == noc]
        ath_count = my_team['Name'].nunique()
        evt_count = my_team['Event'].nunique()

        return ath_count, evt_count, share_total, share_gold, my_medal_cnt, total_medal_cnt

    # 定义分析案例（原有 + 新增 3 个对照组）
    cases = [
        # 你原来的“名帅/效应”案例
        {"Sport": "Cycling Track", "Year": 2008, "NOC": "GBR"},
        {"Sport": "Cycling Track", "Year": 2012, "NOC": "GBR"},
        {"Sport": "Judo", "Year": 2012, "NOC": "JPN"},
        {"Sport": "Judo", "Year": 2016, "NOC": "JPN"},
        {"Sport": "Judo", "Year": 2020, "NOC": "JPN"},
        {"Sport": "Judo", "Year": 2024, "NOC": "JPN"},
        {"Sport": "Volleyball Women", "Year": 2008, "NOC": "USA"},
        {"Sport": "Volleyball Women", "Year": 2008, "NOC": "CHN"},
        {"Sport": "Volleyball Women", "Year": 2016, "NOC": "CHN"},
        {"Sport": "Volleyball Women", "Year": 2016, "NOC": "USA"},

        # ==========================
        # 新增：对照组 3 个案例
        # ==========================
        {"Sport": "Athletics", "Year": 2016, "NOC": "USA"},   # 对比 2012
        {"Sport": "Rowing", "Year": 2012, "NOC": "NZL"},      # 对比 2008
        {"Sport": "Wrestling", "Year": 2016, "NOC": "AZE"},   # 对比 2012
    ]

    results = []
    host_map = {2004: 'GRE', 2008: 'CHN', 2012: 'GBR', 2016: 'BRA', 2020: 'JPN', 2024: 'FRA'}

    for c in cases:
        year = c["Year"]
        sport = c["Sport"]
        noc = c["NOC"]
        lag_year = year - 4

        # 1) 滞后一期
        lag_ath, lag_evt, lag_share_total, lag_share_gold, lag_medals, lag_total_medals = get_stats(
            df, sport, lag_year, noc
        )

        # 2) 当期
        ath, evt, real_total_share, real_gold_share, real_medals, total_medals = get_stats(
            df, sport, year, noc
        )
        is_host = 1 if host_map.get(year) == noc else 0

        # 3) Tobit 预测（你只输出 Total，但 gold 仍用于内部函数参数）
        pred_total, _ = predict_mit_tobit(ath, evt, is_host, lag_share_total, lag_share_gold)

        # 4) Delta：Real - Pred
        delta_total = real_total_share - pred_total

        # 5) 论文用的“投入/效率”诊断
        eff = (real_medals / ath) if ath > 0 else 0
        lag_eff = (lag_medals / lag_ath) if lag_ath > 0 else 0
        eff_ratio = (eff / lag_eff) if lag_eff > 0 else np.nan

        results.append({
            "Sport": sport,
            "Year": year,
            "Lag_Year": lag_year,
            "NOC": noc,

            # 真实与预测（MIT）
            "Real_MIT_Total": round(real_total_share, 6),
            "Pred_MIT_Total": round(pred_total, 6),
            "Delta_MIT_Total": round(delta_total, 6),

            # 诊断字段：奖牌数/队伍规模/效率变化（对照组论证用）
            "Real_Medals": int(real_medals),
            "Lag_Medals": int(lag_medals),
            "Athletes": int(ath),
            "Lag_Athletes": int(lag_ath),
            "Eff_MedalsPerAth": round(eff, 6),
            "Lag_Eff_MedalsPerAth": round(lag_eff, 6),
            "Eff_Ratio": round(eff_ratio, 6) if pd.notna(eff_ratio) else np.nan,

            # 可选：你如果想看模型输入也可以留着
            "Events": int(evt),
            "Lag_Events": int(lag_evt),
        })

    # 导出
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values('Delta_MIT_Total', ascending=False)

    print("-" * 100)
    print("ANALYSIS RESULTS (Delta = Real - Pred)")
    print("Delta 越大：实际表现显著高于模型预测（控制 TeamSize 后仍抬升，可能存在外生效应）")
    print("-" * 100)
    cols_show = [
        "Sport", "Year", "NOC",
        "Real_MIT_Total", "Pred_MIT_Total", "Delta_MIT_Total",
        "Real_Medals", "Lag_Medals",
        "Athletes", "Lag_Athletes",
        "Eff_Ratio"
    ]
    print(df_out[cols_show].to_string(index=False))

    df_out.to_csv("mit_interpolation.csv", index=False, encoding="utf-8-sig")
    print("\n结果已保存至: mit_interpolation.csv")


if __name__ == "__main__":
    run_analysis_and_save()
