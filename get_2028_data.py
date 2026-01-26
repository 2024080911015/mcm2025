import pandas as pd

# 1. 读取原始数据
# 请确保文件名与您上传的文件名一致
file_path = 'modeling_data_maturity.csv'
df = pd.read_csv(file_path)

# 2. 筛选出 2024 年的数据
df_2024 = df[df['Year'] == 2024]

# 3. 简单查看一下结果 (可选)
print("2024年数据概览：")
print(df_2024.head())
print(f"总共有 {len(df_2024)} 个国家/地区的数据。")

# 4. 导出为新的 CSV 文件
output_file = 'data_2028.csv'
df_2024.to_csv(output_file, index=False)

print(f"文件已成功导出为: {output_file}")