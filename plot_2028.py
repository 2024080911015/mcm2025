import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map

# 1. 读取数据
# 请确保 csv 文件在当前目录下，或者修改为绝对路径
file_path = "prediction_2028_final.csv"
df = pd.read_csv(file_path)

# 2. 数据处理
# 将预测份额转换为百分比，保留两位小数
df['Share_Pct'] = round(df['Predicted_Medal_Share'] * 100, 2)

# 3. 国家名称映射 (重要)
# 地图通常使用标准英文名，而奥运会数据常使用IOC名称
# 这种不一致会导致部分国家在地图上无法着色，需要手动修正
name_mapping = {
    "Great Britain": "United Kingdom",
    "South Korea": "Korea",
    "North Korea": "Dem. Rep. Korea",
    "Chinese Taipei": "Taiwan",
    "United States": "United States",  # 确保一致
    # 如有其他国家不显示，可在此添加映射
}

# 应用名称映射，如果没在字典里则保持原名
df['Map_Name'] = df['Country_Name'].replace(name_mapping)

# 准备 pyecharts 需要的数据格式： [("Country Name", Value), ...]
data_pair = [list(z) for z in zip(df['Map_Name'], df['Share_Pct'])]

# 4. 创建地图
# 根据数据最大值（美国约17.6%）设置 max_ 为 20
max_value = 20

world_map = (
    Map(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color="#FFFFFF",renderer="svg"))
    .add(
        series_name="预测奖牌占比(%)",
        data_pair=data_pair,
        maptype="world",
        is_map_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
        itemstyle_opts=opts.ItemStyleOpts(
            border_width=0.5,
            border_color="rgba(0,0,0,0.2)"
        ),
        zoom=1.2,
        emphasis_label_opts=opts.LabelOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{b}: {c}%"
        )
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="2028奥运会各国奖牌预测占比",
            subtitle="数据来源：Prediction 2028",
            pos_left="center",
            title_textstyle_opts=opts.TextStyleOpts(font_size=20)
        ),
        visualmap_opts=opts.VisualMapOpts(
            max_=max_value,
            min_=0,
            is_piecewise=False,
            # 使用你指定的颜色渐变：蓝 -> 黄 -> 红
            range_color=["#50a3ba", "#eac736", "#d94e5d"],
            orient="horizontal",
            pos_left="center",
            pos_bottom="10%"
        ),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

# 5. 渲染地图
output_file = "2028_Olympic_Prediction_Map.html"
world_map.render(output_file)
print(f"地图已生成：{output_file}")