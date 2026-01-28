import pandas as pd
import numpy as np
from scipy.stats import norm

# =========================================================
# 0) 配置区
# =========================================================
ATHLETES_CSV = "summerOly_athletes.csv"
COEFFS_CSV = "sport_tobit_coeffs.csv"
OUT_CSV = "mit_timeseries_2000_2024.csv"

# 只算这些 sport（与你例子一致）
SPORTS = [
    "Cycling Track",
    "Judo",
    "Volleyball Women",
    "Athletics",
    "Rowing",
    "Wrestling",
]

# 只输出这些国家（与你例子一致）
NOCS = ["GBR", "JPN", "USA", "CHN", "NZL", "AZE"]

# 2000-2024 每届夏奥
YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]

# 东道主映射（补全 2000）
HOST_MAP = {2000: "AUS", 2004: "GRE", 2008: "CHN", 2012: "GBR", 2016: "BRA", 2020: "JPN", 2024: "FRA"}


# =========================================================
# 1) sport 过滤逻辑（保持你之前口径）
# =========================================================
def filter_sport_df(df, sport_label):
    if sport_label == "Cycling Track":
        temp = df[df["Sport"] == "Cycling"].copy()
        exclude = ["Mountainbike", "Road Race", "Individual Time Trial", "BMX"]
        pat = "|".join(exclude)
        temp = temp[~temp["Event"].astype(str).str.contains(pat, case=False, regex=True)]
        return temp

    if sport_label == "Volleyball Women":
        temp = df[
            (df["Sport"] == "Volleyball")
            & (df["Event"].astype(str).str.contains("Women", case=False, regex=True))
        ].copy()
        return temp

    return df[df["Sport"] == sport_label].copy()


# =========================================================
# 2) 构造 sport-level 面板 (Year,NOC)
#    MIT_Total = My_Medals / Total_Medals_In_SportYear
#    X = [1, lnAth, lnEvt, host, lag(MIT_Total at year-4)]
# =========================================================
def build_sport_panel(df, sport_label, host_map):
    temp = filter_sport_df(df, sport_label)

    # 基础特征
    ath = temp.groupby(["Year", "NOC"])["Name"].nunique().rename("Athlete_Count")
    evt = temp.groupby(["Year", "NOC"])["Event"].nunique().rename("Events_Participated_Count")
    base = pd.concat([ath, evt], axis=1).fillna(0).reset_index()

    # 奖牌计数（国家-事件-奖牌 唯一）
    medaled = temp[temp["Medal"].isin(["Gold", "Silver", "Bronze"])].copy()
    all_medals = medaled.drop_duplicates(subset=["Year", "Event", "Medal", "NOC"])

    denom = all_medals.groupby("Year").size().rename("Total_Medals_In_SportYear").reset_index()
    numer = all_medals.groupby(["Year", "NOC"]).size().rename("My_Medals").reset_index()

    panel = base.merge(denom, on="Year", how="left").merge(numer, on=["Year", "NOC"], how="left")
    panel["Total_Medals_In_SportYear"] = panel["Total_Medals_In_SportYear"].fillna(0).astype(int)
    panel["My_Medals"] = panel["My_Medals"].fillna(0).astype(int)

    panel["MIT_Total"] = np.where(
        panel["Total_Medals_In_SportYear"] > 0,
        panel["My_Medals"] / panel["Total_Medals_In_SportYear"],
        0.0,
    )

    panel["Is_Host"] = panel.apply(
        lambda r: 1.0 if host_map.get(int(r["Year"])) == r["NOC"] else 0.0, axis=1
    )

    panel["Ln_Athletes"] = np.log(panel["Athlete_Count"].replace(0, np.nan)).fillna(0.0)
    panel["Ln_Events"] = np.log(panel["Events_Participated_Count"].replace(0, np.nan)).fillna(0.0)

    # lag: 严格 year-4 的真实 MIT_Total
    lag_df = panel[["Year", "NOC", "MIT_Total"]].copy()
    lag_df["Year"] = lag_df["Year"] + 4
    lag_df = lag_df.rename(columns={"MIT_Total": "Lagged_MIT_Total"})
    panel = panel.merge(lag_df, on=["Year", "NOC"], how="left")
    panel["Lagged_MIT_Total"] = panel["Lagged_MIT_Total"].fillna(0.0)

    panel["Sport"] = sport_label
    return panel


# =========================================================
# 3) Tobit 的“截断在0”的期望（解析版）
#    E[y|x] = Phi(z)*mu + sigma*phi(z) , z = mu/sigma
# =========================================================
def tobit_expected(mu, sigma):
    z = mu / sigma
    return norm.cdf(z) * mu + sigma * norm.pdf(z)


# =========================================================
# 4) 用已拟合参数对 sport-year 做预测，并在 sport-year 内归一化
#    返回每个 NOC 的 Pred_MIT_Total（预测份额）
# =========================================================
def predict_sport_year(panel_sport, params, year):
    """
    params: [Intercept, b_lnAth, b_lnEvt, b_host, b_lag, sigma]
    """
    sub = panel_sport[panel_sport["Year"] == year].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Sport", "Year", "NOC", "Pred_MIT_Total"])

    intercept, b_lnAth, b_lnEvt, b_host, b_lag, sigma = params

    mu = (
        intercept
        + b_lnAth * sub["Ln_Athletes"].values
        + b_lnEvt * sub["Ln_Events"].values
        + b_host * sub["Is_Host"].values
        + b_lag * sub["Lagged_MIT_Total"].values
    )

    # Tobit 截断期望，天然>=0
    raw = tobit_expected(mu, sigma)
    raw = np.maximum(0.0, raw)

    # sport-year 内归一化（使份额和=1）
    s = raw.sum()
    pred_share = raw / s if s > 0 else np.zeros_like(raw)

    out = sub[["Sport", "Year", "NOC"]].copy()
    out["Pred_MIT_Total"] = pred_share
    return out


# =========================================================
# 5) 主程序：生成 2000-2024（每届）指定 sport×NOC 的 Real/Pred MIT
# =========================================================
def main():
    # --- 读取数据 ---
    df = pd.read_csv(ATHLETES_CSV)

    # --- 读取已拟合参数 ---
    coef = pd.read_csv(COEFFS_CSV)

    # 只取我们关心的 sports，并确保有参数
    coef = coef[coef["Sport"].isin(SPORTS)].copy()
    if coef.empty:
        raise ValueError("sport_tobit_coeffs.csv 里找不到指定 SPORTS 的拟合结果。")

    # sport -> params
    # （如果同一个 sport 有多行，优先 Converged=True 且 NegLogL 更小的那行）
    models = {}
    for sp in SPORTS:
        sub = coef[coef["Sport"] == sp].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by=["Converged", "NegLogL"], ascending=[False, True])
        row = sub.iloc[0]
        params = np.array(
            [row["Intercept"], row["b_lnAth"], row["b_lnEvt"], row["b_host"], row["b_lag"], row["sigma"]],
            dtype=float,
        )
        models[sp] = params

    missing = [sp for sp in SPORTS if sp not in models]
    if missing:
        print("警告：以下 sport 在系数文件里没有可用参数，将跳过：", missing)

    # --- 构造每个 sport 的 panel（全量年份，用于 lag 计算）---
    sport_panels = {}
    for sp in SPORTS:
        if sp in models:
            sport_panels[sp] = build_sport_panel(df, sp, HOST_MAP)

    # --- 逐 sport×year 预测并拼接 ---
    pred_rows = []
    for sp in SPORTS:
        if sp not in models:
            continue
        panel_sp = sport_panels[sp]
        params = models[sp]
        for y in YEARS:
            pred_rows.append(predict_sport_year(panel_sp, params, y))

    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    # --- 真实 MIT（与你 get_real_mit_total 口径一致：My/Total, sport-year 内）---
    real_rows = []
    for sp in SPORTS:
        temp = filter_sport_df(df, sp)
        for y in YEARS:
            tmpy = temp[temp["Year"] == y]
            if tmpy.empty:
                continue
            medaled = tmpy[tmpy["Medal"].isin(["Gold", "Silver", "Bronze"])]
            all_medals = medaled.drop_duplicates(subset=["Event", "Medal", "NOC"])
            total_count = len(all_medals)
            if total_count == 0:
                continue
            numer = all_medals.groupby("NOC").size().rename("My_Medals").reset_index()
            numer["Real_MIT_Total"] = numer["My_Medals"] / total_count
            numer["Sport"] = sp
            numer["Year"] = y
            real_rows.append(numer[["Sport", "Year", "NOC", "Real_MIT_Total"]])

    real_df = pd.concat(real_rows, ignore_index=True) if real_rows else pd.DataFrame()

    # --- 合并 Real + Pred ---
    out = real_df.merge(pred_df, on=["Sport", "Year", "NOC"], how="left")

    # 只保留指定 NOC
    out = out[out["NOC"].isin(NOCS)].copy()

    # 排序：按 Sport, NOC, Year
    out = out.sort_values(by=["Sport", "NOC", "Year"]).reset_index(drop=True)

    # 保存
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"完成：已输出 {OUT_CSV}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
