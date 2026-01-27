import pandas as pd

# =============== 1) 真实 MIT（来自我已查到的公开奖牌表/奖牌归属） ===============
actual_rows = [
    # Cycling Track
    {"Sport":"Cycling Track","Year":2008,"NOC":"GBR","Gold":7,"Silver":3,"Bronze":2,"Total":12,"Total_Medals_In_Sport":30,"Total_Golds_In_Sport":10},
    {"Sport":"Cycling Track","Year":2012,"NOC":"GBR","Gold":7,"Silver":1,"Bronze":1,"Total":9,"Total_Medals_In_Sport":31,"Total_Golds_In_Sport":10},

    # Judo
    {"Sport":"Judo","Year":2012,"NOC":"JPN","Gold":4,"Silver":1,"Bronze":2,"Total":7,"Total_Medals_In_Sport":56,"Total_Golds_In_Sport":14},
    {"Sport":"Judo","Year":2016,"NOC":"JPN","Gold":3,"Silver":1,"Bronze":8,"Total":12,"Total_Medals_In_Sport":56,"Total_Golds_In_Sport":14},
    {"Sport":"Judo","Year":2020,"NOC":"JPN","Gold":9,"Silver":2,"Bronze":1,"Total":12,"Total_Medals_In_Sport":60,"Total_Golds_In_Sport":15},
    {"Sport":"Judo","Year":2024,"NOC":"JPN","Gold":3,"Silver":2,"Bronze":3,"Total":8,"Total_Medals_In_Sport":60,"Total_Golds_In_Sport":15},

    # Volleyball Women (每届该项目只产生 1金1银1铜，共3枚奖牌)
    {"Sport":"Volleyball Women","Year":2008,"NOC":"USA","Gold":0,"Silver":1,"Bronze":0,"Total":1,"Total_Medals_In_Sport":3,"Total_Golds_In_Sport":1},
    {"Sport":"Volleyball Women","Year":2008,"NOC":"CHN","Gold":0,"Silver":0,"Bronze":1,"Total":1,"Total_Medals_In_Sport":3,"Total_Golds_In_Sport":1},
    {"Sport":"Volleyball Women","Year":2016,"NOC":"CHN","Gold":1,"Silver":0,"Bronze":0,"Total":1,"Total_Medals_In_Sport":3,"Total_Golds_In_Sport":1},
    {"Sport":"Volleyball Women","Year":2016,"NOC":"USA","Gold":0,"Silver":0,"Bronze":1,"Total":1,"Total_Medals_In_Sport":3,"Total_Golds_In_Sport":1},
]

df_actual = pd.DataFrame(actual_rows)
df_actual["Real_MIT_Total"] = df_actual["Total"] / df_actual["Total_Medals_In_Sport"]
df_actual["Real_MIT_Gold"]  = df_actual["Gold"]  / df_actual["Total_Golds_In_Sport"]

# 如果你想保存真实MIT：
df_actual.to_csv("real_mit_from_public_results.csv", index=False, encoding="utf-8-sig")


# =============== 2) 读取你的预测 MIT ===============
# 你把文件名换成你自己的
df_pred = pd.read_csv("pred_mit.csv")

required_cols = {"Sport","Year","NOC","Pred_MIT_Total"}
missing = required_cols - set(df_pred.columns)
if missing:
    raise ValueError(f"pred_mit.csv 缺少必要列: {missing}")

# =============== 3) 合并 + 计算插值（预测 - 真实） ===============
df = df_pred.merge(
    df_actual[["Sport","Year","NOC","Real_MIT_Total","Real_MIT_Gold"]],
    on=["Sport","Year","NOC"],
    how="left"
)

# 插值/残差：pred - real
df["Delta_MIT_Total"] = df["Pred_MIT_Total"] - df["Real_MIT_Total"]

if "Pred_MIT_Gold" in df.columns:
    df["Delta_MIT_Gold"] = df["Pred_MIT_Gold"] - df["Real_MIT_Gold"]

# 你可以按绝对误差排序，快速看到“教练效应/突变”候选点
df["AbsDelta_Total"] = df["Delta_MIT_Total"].abs()
sort_cols = ["AbsDelta_Total"]
df_out = df.sort_values(sort_cols, ascending=False)

df_out.to_csv("mit_interpolation.csv", index=False, encoding="utf-8-sig")
print("已输出: mit_interpolation.csv")
