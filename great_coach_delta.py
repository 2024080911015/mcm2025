import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


# =========================================================
# 1) Tobit 负对数似然（同你原逻辑）
# =========================================================
def weighted_tobit_log_likelihood(params, X, y, weights):
    beta = params[:-1]
    sigma = params[-1]
    if sigma <= 1e-6:
        return 1e10

    mu = X @ beta
    censored = (y <= 0)
    uncensored = ~censored

    prob_cens = norm.cdf(-mu[censored] / sigma)
    ll_cens = weights[censored] * np.log(np.clip(prob_cens, 1e-15, 1.0))

    z = (y[uncensored] - mu[uncensored]) / sigma
    ll_unc = weights[uncensored] * (-np.log(sigma) + norm.logpdf(z))

    return -(ll_cens.sum() + ll_unc.sum())


# =========================================================
# 2) sport 过滤逻辑（兼容 Cycling Track / Volleyball Women）
# =========================================================
def filter_sport_df(df, sport_label):
    if sport_label == "Cycling Track":
        temp = df[df["Sport"] == "Cycling"].copy()
        exclude = ['Mountainbike', 'Road Race', 'Individual Time Trial', 'BMX']
        pat = "|".join(exclude)
        temp = temp[~temp["Event"].astype(str).str.contains(pat, case=False, regex=True)]
        return temp

    if sport_label == "Volleyball Women":
        temp = df[(df["Sport"] == "Volleyball") &
                  (df["Event"].astype(str).str.contains("Women", case=False, regex=True))].copy()
        return temp

    return df[df["Sport"] == sport_label].copy()


# =========================================================
# 3) 从 summerOly_athletes 构造 sport-level 面板 (Year,NOC)
#    y = MIT_Total(sport,year,noc)
#    X = [1, lnAth, lnEvt, host, lag(MIT_Total at year-4)]
# =========================================================
def build_sport_panel(df, sport_label, host_map):
    temp = filter_sport_df(df, sport_label)

    ath = temp.groupby(["Year", "NOC"])["Name"].nunique().rename("Athlete_Count")
    evt = temp.groupby(["Year", "NOC"])["Event"].nunique().rename("Events_Participated_Count")
    base = pd.concat([ath, evt], axis=1).fillna(0).reset_index()

    medaled = temp[temp["Medal"].isin(["Gold", "Silver", "Bronze"])].copy()
    all_medals = medaled.drop_duplicates(subset=["Year", "Event", "Medal", "NOC"])

    denom = all_medals.groupby("Year").size().rename("Total_Medals_In_SportYear").reset_index()
    numer = all_medals.groupby(["Year", "NOC"]).size().rename("My_Medals").reset_index()

    panel = base.merge(denom, on="Year", how="left").merge(numer, on=["Year", "NOC"], how="left")
    panel["Total_Medals_In_SportYear"] = panel["Total_Medals_In_SportYear"].fillna(0).astype(int)
    panel["My_Medals"] = panel["My_Medals"].fillna(0).astype(int)

    panel["MIT_Total"] = np.where(panel["Total_Medals_In_SportYear"] > 0,
                                 panel["My_Medals"] / panel["Total_Medals_In_SportYear"],
                                 0.0)

    panel["Is_Host"] = panel.apply(
        lambda r: 1.0 if host_map.get(int(r["Year"])) == r["NOC"] else 0.0, axis=1
    )

    panel["Ln_Athletes"] = np.log(panel["Athlete_Count"].replace(0, np.nan)).fillna(0.0)
    panel["Ln_Events"] = np.log(panel["Events_Participated_Count"].replace(0, np.nan)).fillna(0.0)

    # lag: 严格 year-4
    lag_df = panel[["Year", "NOC", "MIT_Total"]].copy()
    lag_df["Year"] = lag_df["Year"] + 4
    lag_df = lag_df.rename(columns={"MIT_Total": "Lagged_MIT_Total"})
    panel = panel.merge(lag_df, on=["Year", "NOC"], how="left")
    panel["Lagged_MIT_Total"] = panel["Lagged_MIT_Total"].fillna(0.0)

    panel["Sport"] = sport_label
    return panel


# =========================================================
# 4) 单 sport 拟合 Tobit（返回 scipy optimize res）
# =========================================================
def fit_tobit_for_sport(panel):
    y = panel["MIT_Total"].values.astype(float)
    X = panel[["Ln_Athletes", "Ln_Events", "Is_Host", "Lagged_MIT_Total"]].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])
    weights = np.ones(len(y))

    guesses = [
        [-0.05, 0.01, 0.01, 0.02, 0.5, 0.05],
        [-0.15, 0.005, 0.005, 0.01, 0.3, 0.02],
        [-0.02, 0.02, 0.02, 0.05, 0.8, 0.10],
    ]

    best = None
    for g in guesses:
        res = minimize(
            weighted_tobit_log_likelihood,
            np.array(g, dtype=float),
            args=(X, y, weights),
            method="L-BFGS-B",
            bounds=[(None, None)] * 5 + [(1e-6, None)],
            options={"maxiter": 3000, "ftol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res
        if res.success:
            best = res
            break

    return best


# =========================================================
# 5) cases 的 Real MIT 计算（同你原版口径）
# =========================================================
def get_real_mit_total(df, sport_label, year, noc):
    temp = filter_sport_df(df, sport_label)
    temp = temp[temp["Year"] == year]

    medaled = temp[temp["Medal"].isin(["Gold", "Silver", "Bronze"])]
    all_medals = medaled.drop_duplicates(subset=["Event", "Medal", "NOC"])
    my_medals = all_medals[all_medals["NOC"] == noc]

    total_count = len(all_medals)
    my_count = len(my_medals)
    return (my_count / total_count) if total_count > 0 else 0.0


# =========================================================
# 6) Monte Carlo：对一个 (sport, year) 同时模拟所有 NOC → 归一化 → 抽取 case NOC
#     输出 Pred_sim 分布，从而得到 Pred_CI 和 Delta_CI
# =========================================================
def mc_predict_case_delta(panel_sport, params, year, case_noc, n_sims=10000, batch=2000, seed=42):
    """
    panel_sport: build_sport_panel() 的返回值（包含该 sport 全年份）
    params: [Intercept, lnAth, lnEvt, host, lag, sigma]
    """
    rng = np.random.default_rng(seed)

    sub = panel_sport[panel_sport["Year"] == year].copy()
    if sub.empty:
        return np.nan, (np.nan, np.nan), (np.nan, np.nan)

    # 设计矩阵（不含 sigma）
    X = sub[["Ln_Athletes", "Ln_Events", "Is_Host", "Lagged_MIT_Total"]].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])

    beta = params[:-1].astype(float)
    sigma = float(params[-1])
    mu = X @ beta  # shape (n_noc,)

    nocs = sub["NOC"].astype(str).tolist()
    if case_noc not in nocs:
        return np.nan, (np.nan, np.nan), (np.nan, np.nan)

    idx = nocs.index(case_noc)

    # 分批模拟，避免大矩阵占内存
    sims_case = []
    remaining = n_sims
    while remaining > 0:
        b = min(batch, remaining)
        remaining -= b

        shocks = rng.normal(loc=0.0, scale=sigma, size=(b, len(mu)))   # (b, n_noc)
        shares = mu.reshape(1, -1) + shocks                              # (b, n_noc)
        shares = np.maximum(0.0, shares)                                 # 截断负值

        row_sum = shares.sum(axis=1, keepdims=True)
        # 归一化：每次模拟总和=1（若全0，则保持0）
        shares = np.where(row_sum > 0, shares / row_sum, 0.0)

        sims_case.append(shares[:, idx])

    pred_sim = np.concatenate(sims_case)  # length n_sims

    # 预测份额的均值与区间
    pred_mean = float(np.mean(pred_sim))
    pred_ci = (float(np.percentile(pred_sim, 2.5)), float(np.percentile(pred_sim, 97.5)))

    # Delta 的区间要等拿到 real 后再算，这里先返回 pred_sim 以便外层算
    return pred_mean, pred_ci, pred_sim


# =========================================================
# 7) 主程序：一键拟合所有 sport + MC 预测 cases + 输出原版格式 + Delta CI
# =========================================================
def run_all_fit_predict_mc():
    df = pd.read_csv("summerOly_athletes.csv")

    host_map = {2004: "GRE", 2008: "CHN", 2012: "GBR", 2016: "BRA", 2020: "JPN", 2024: "FRA"}

    # 你的 cases（可继续加）
    cases = [
        {"Sport": "Cycling Track", "Year": 2008, "NOC": "GBR"},
        #{"Sport": "Cycling Track", "Year": 2012, "NOC": "GBR"},
        {"Sport": "Judo", "Year": 2012, "NOC": "JPN"},
        {"Sport": "Judo", "Year": 2016, "NOC": "JPN"},
        {"Sport": "Judo", "Year": 2020, "NOC": "JPN"},
        #{"Sport": "Judo", "Year": 2024, "NOC": "JPN"},
        {"Sport": "Volleyball Women", "Year": 2008, "NOC": "USA"},
        {"Sport": "Volleyball Women", "Year": 2008, "NOC": "CHN"},
        {"Sport": "Volleyball Women", "Year": 2016, "NOC": "CHN"},
        {"Sport": "Volleyball Women", "Year": 2016, "NOC": "USA"},

        # 对照组
        {"Sport": "Athletics", "Year": 2016, "NOC": "USA"},
        {"Sport": "Rowing", "Year": 2012, "NOC": "NZL"},
        {"Sport": "Wrestling", "Year": 2016, "NOC": "AZE"},
    ]

    # “所有 sport”：原始 Sport 列 + 你自定义的两个标签
    sport_labels = sorted(set(df["Sport"].dropna().astype(str).unique()).union({"Cycling Track", "Volleyball Women"}))

    # 先构造每个 sport 的 panel（避免重复计算）
    sport_panels = {}
    for sp in sport_labels:
        sport_panels[sp] = build_sport_panel(df, sp, host_map)

    # 拟合每个 sport 的 Tobit
    models = {}  # sp -> params
    coef_rows = []

    print("\n" + "=" * 80)
    print("FITTING TOBIT MODELS BY SPORT (from summerOly_athletes.csv) ...")
    print("=" * 80)

    for sp in sport_labels:
        panel = sport_panels[sp]
        if panel.shape[0] < 30:  # 样本太小跳过（你可调小/删掉）
            continue

        res = fit_tobit_for_sport(panel)
        if res.x is None:
            continue

        models[sp] = res.x

        coef_rows.append({
            "Sport": sp,
            "Intercept": res.x[0],
            "b_lnAth": res.x[1],
            "b_lnEvt": res.x[2],
            "b_host": res.x[3],
            "b_lag": res.x[4],
            "sigma": res.x[5],
            "Converged": bool(res.success),
            "NegLogL": float(res.fun),
            "n_obs": int(panel.shape[0])
        })

    pd.DataFrame(coef_rows).to_csv("sport_tobit_coeffs.csv", index=False, encoding="utf-8-sig")
    print("已保存：sport_tobit_coeffs.csv")

    # ====== 对 cases 做 Monte Carlo 预测并输出 Delta CI ======
    N_SIMS = 10000
    BATCH = 2000
    SEED = 2026

    results = []

    print("\n" + "=" * 80)
    print(f"MONTE CARLO SIMULATION: N_SIMS={N_SIMS} (normalize within sport-year)")
    print("=" * 80)

    for c in cases:
        sp = c["Sport"]
        year = int(c["Year"])
        noc = c["NOC"]

        real = get_real_mit_total(df, sp, year, noc)


        if sp not in models:
            results.append({
                "Sport": sp,
                "Year": year,
                "NOC": noc,
                "Real_MIT_Total": round(real, 4),
                "Pred_MIT_Total": np.nan,
                "Delta_MIT_Total": np.nan,
                "Pred_95_CI": np.nan,
                "Delta_95_CI": np.nan
            })
            continue

        panel = sport_panels[sp]
        pred_mean, pred_ci, pred_sim = mc_predict_case_delta(
            panel, models[sp], year, noc, n_sims=N_SIMS, batch=BATCH, seed=SEED
        )

        if isinstance(pred_sim, float) and np.isnan(pred_sim):
            # 兜底
            results.append({
                "Sport": sp,
                "Year": year,
                "NOC": noc,
                "Real_MIT_Total": round(real, 4),
                "Pred_MIT_Total": np.nan,
                "Delta_MIT_Total": np.nan,
                "Pred_95_CI": np.nan,
                "Delta_95_CI": np.nan
            })
            continue

        delta_sim = real - pred_sim
        p_pos = float(np.mean(delta_sim > 0))  # P(Delta > 0)
        p_neg = float(np.mean(delta_sim < 0))  # P(Delta < 0)
        delta_ci = (float(np.percentile(delta_sim, 2.5)), float(np.percentile(delta_sim, 97.5)))
        delta_point = real - pred_mean

        # ===== 新增指标：delta/(MIT-delta) = (real - pred) / pred =====
        denom = pred_mean  # MIT - delta = pred
        delta_ratio = (delta_point / denom) if denom > 1e-12 else np.nan
        results.append({
            "Sport": sp,
            "Year": year,
            "NOC": noc,
            "Real_MIT_Total": round(real, 4),
            "Pred_MIT_Total": round(pred_mean, 4),
            "Delta_MIT_Total": round(real - pred_mean, 4),
            "Increase_Percent": round(delta_ratio, 4),
            "Pred_95_CI": f"[{pred_ci[0]:.4f}, {pred_ci[1]:.4f}]",
            "Delta_95_CI": f"[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]",
            "Pr_Delta_Pos": round(p_pos, 4),
            "Pr_Delta_Neg": round(p_neg, 4)
        })

    df_out = pd.DataFrame(results).sort_values("NOC", ascending=False)

    print("\n" + "-" * 80)
    print("ANALYSIS RESULTS (Delta = Real - Pred_mean) + Delta_95_CI")
    print("-" * 80)
    print(df_out[["Sport","Year","NOC","Real_MIT_Total","Pred_MIT_Total","Delta_MIT_Total","Delta_95_CI"]].to_string(index=False))

    df_out.to_csv("mit_interpolation.csv", index=False, encoding="utf-8-sig")
    print("\n结果已保存至: mit_interpolation.csv")


if __name__ == "__main__":
    run_all_fit_predict_mc()
