import pandas as pd
import numpy as np

# =========================
# 1) 读入原始数据
# =========================
# 把这里改成你的原始csv文件路径
INPUT_CSV = r"medals_2024_country_event.csv"
OUTPUT_CSV = r"country_sport_CR_RCA.csv"

df = pd.read_csv(INPUT_CSV)

# -------------------------
# 可选：基础列检查（防止列名不一致）
# -------------------------
required_cols = {"NOC", "Sport", "Total"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"缺少必要列: {missing}. 你当前列名为: {list(df.columns)}")

# 如果你的 Total 是字符串/空值，先转数值
df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)

# 若金银铜列存在，也转数值（可选，主要用于输出附带信息）
for c in ["Gold", "Silver", "Bronze"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# =========================
# 2) 聚合到 国家(NOC)-项目(Sport) 层面
#    Medals_{c,s} = sum(Total) over events
# =========================
agg_dict = {"Total": "sum"}
if "Gold" in df.columns:   agg_dict["Gold"] = "sum"
if "Silver" in df.columns: agg_dict["Silver"] = "sum"
if "Bronze" in df.columns: agg_dict["Bronze"] = "sum"

agg = (
    df.groupby(["NOC", "Sport"], as_index=False)
      .agg(agg_dict)
      .rename(columns={
          "Total": "Medals_cs",
          "Gold": "Gold_cs",
          "Silver": "Silver_cs",
          "Bronze": "Bronze_cs"
      })
)

# =========================
# 3) 计算国家总奖牌 TotalMedals_c
# =========================
country_totals = (
    agg.groupby("NOC", as_index=False)
       .agg(TotalMedals_c=("Medals_cs", "sum"))
)

# =========================
# 4) 计算世界层面的 WorldMedals_s 和 WorldTotalMedals
# =========================
sport_totals = (
    agg.groupby("Sport", as_index=False)
       .agg(WorldMedals_s=("Medals_cs", "sum"))
)

WorldTotalMedals = float(agg["Medals_cs"].sum())

# =========================
# 5) 合并并计算 CR 与 RCA
#    CR_{c,s} = Medals_{c,s} / TotalMedals_c
#    RCA_{c,s} = (Medals_{c,s}/TotalMedals_c) / (WorldMedals_s/WorldTotalMedals)
# =========================
out = (
    agg.merge(country_totals, on="NOC", how="left")
       .merge(sport_totals, on="Sport", how="left")
)

# 绝对贡献率 CR
out["CR_c_s"] = np.where(
    out["TotalMedals_c"] > 0,
    out["Medals_cs"] / out["TotalMedals_c"],
    np.nan
)

# 世界该项目占比
out["WorldShare_s"] = np.where(
    out["WorldMedals_s"] > 0,
    out["WorldMedals_s"] / WorldTotalMedals,
    np.nan
)

# 显性比较优势 RCA
out["RCA_c_s"] = np.where(
    out["WorldShare_s"] > 0,
    out["CR_c_s"] / out["WorldShare_s"],
    np.nan
)

# 可选：优势标记
out["RCA_gt_1"] = out["RCA_c_s"] > 1
out["RCA_gt_5"] = out["RCA_c_s"] > 5

# =========================
# 6) 排序 & 输出
# =========================
# 排序逻辑：同一国家下，让“核心项目”(CR高)排在前面，其次看RCA
out = out.sort_values(["NOC", "CR_c_s", "RCA_c_s"], ascending=[True, False, False])

# 输出列顺序（你也可以按自己论文需求改）
cols = [
    "NOC", "Sport",
    "Medals_cs", "TotalMedals_c",
    "WorldMedals_s", "WorldShare_s",
    "CR_c_s", "RCA_c_s",
    "RCA_gt_1", "RCA_gt_5"
]

# 如果有金银铜列就附加
for c in ["Gold_cs", "Silver_cs", "Bronze_cs"]:
    if c in out.columns:
        cols.append(c)

out = out[cols].reset_index(drop=True)

out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"已生成: {OUTPUT_CSV}")
print(out.head(10))
