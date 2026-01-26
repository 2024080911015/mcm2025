import numpy as np
import pandas as pd

# ==========================================
# 1. Early Data (早期)
# ==========================================
early_data = pd.read_csv("modeling_data_early.csv")
# print(early_data)
early_country = early_data['NOC']
early_medal_source = early_data['Total_Medals']  # 修改点1：给原始数据换个名，避免冲突

early_country_count = {}
early_medal_sum = {}  # 修改点2：新建一个字典来存奖牌总数

for i, country in enumerate(early_country):
    year = early_data['Year'][i]
    medal_val = early_medal_source[i]  # 获取当前这一行的奖牌数

    if year not in early_country_count:
        early_country_count[year] = 0
        early_medal_sum[year] = 0

    early_country_count[year] += 1
    early_medal_sum[year] += medal_val  # 修改点3：这里要加具体的奖牌数值

df = pd.DataFrame(list(early_country_count.items()), columns=['Year', 'Country_Count'])
df2 = pd.DataFrame(list(early_medal_sum.items()), columns=['Year', 'Medals_count'])

early_count_av = df['Country_Count'].mean()
early_medal_av = df2['Medals_count'].mean()

print("--- Early ---")
print(early_count_av)
print(early_medal_av)

# ==========================================
# 2. Formation Data (成长期)
# ==========================================
formation_data = pd.read_csv("modeling_data_formation.csv")
formation_country = formation_data['NOC']
formation_medal_source = formation_data['Total_Medals']

formation_country_count = {}
formation_medal_sum = {}

for i, country in enumerate(formation_country):
    year = formation_data['Year'][i]
    medal_val = formation_medal_source[i]

    if year not in formation_country_count:
        formation_country_count[year] = 0
        formation_medal_sum[year] = 0

    formation_country_count[year] += 1
    formation_medal_sum[year] += medal_val

df = pd.DataFrame(list(formation_country_count.items()), columns=['Year', 'Country_Count'])
df2 = pd.DataFrame(list(formation_medal_sum.items()), columns=['Year', 'Medals_count'])

formation_count_av = df['Country_Count'].mean()
formation_medal_av = df2['Medals_count'].mean()

print("\n--- Formation ---")
print(formation_count_av)
print(formation_medal_av)

# ==========================================
# 3. Maturity Data (成熟期)
# ==========================================
maturity_data = pd.read_csv("modeling_data_maturity.csv")
maturity_country = maturity_data['NOC']
maturity_medal_source = maturity_data['Total_Medals']

maturity_country_count = {}
maturity_medal_sum = {}

for i, country in enumerate(maturity_country):
    year = maturity_data['Year'][i]
    medal_val = maturity_medal_source[i]

    if year not in maturity_country_count:
        maturity_country_count[year] = 0
        maturity_medal_sum[year] = 0

    maturity_country_count[year] += 1
    maturity_medal_sum[year] += medal_val

df = pd.DataFrame(list(maturity_country_count.items()), columns=['Year', 'Count'])
df2 = pd.DataFrame(list(maturity_medal_sum.items()), columns=['Year', 'Medals_count'])

maturity_count_av = df['Count'].mean()
maturity_medal_av = df2['Medals_count'].mean()

print("\n--- Maturity ---")
print(maturity_count_av)
print(maturity_medal_av)

early_dp=(np.abs((early_count_av-maturity_count_av)/maturity_count_av)+np.abs(early_medal_av-maturity_medal_av)/maturity_medal_av)/2
early_weight=1/(1+early_dp)
print(early_weight)

formation_dp = (np.abs((formation_count_av - maturity_count_av) / maturity_count_av) +
                np.abs((formation_medal_av - maturity_medal_av) / maturity_medal_av)) / 2
formation_weight = 1 / (1 + formation_dp)
print("Formation Weight:", formation_weight)

# 3. Maturity Weight (补全部分)
# 计算逻辑：Maturity 相比于 Maturity 自己，差异度应该是 0，权重应该是 1
maturity_dp = (np.abs((maturity_count_av - maturity_count_av) / maturity_count_av) +
               np.abs((maturity_medal_av - maturity_medal_av) / maturity_medal_av)) / 2
maturity_weight = 1 / (1 + maturity_dp)
print("Maturity Weight:", maturity_weight)