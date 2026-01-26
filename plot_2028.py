import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map, Page

# 1. 读取数据
file_path = "prediction_2028_final.csv"
df = pd.read_csv(file_path)

# 2. 数据预处理
# 确保数值为整数
df['Total_Medals'] = df['Predicted_Total_Medals'].astype(int)
df['Gold_Medals'] = df['Predicted_Gold'].astype(int)

# 3. 国家名称映射 (修正版，确保半岛两国显示正确)
name_mapping = {
    "Great Britain": "United Kingdom",
    #"South Korea": "Korea",
    #"North Korea": "Dem. Rep. Korea",
    "Chinese Taipei": "Taiwan",
    "United States": "United States",
}
df['Map_Name'] = df['Country_Name'].replace(name_mapping)

# 准备数据对
data_total = [list(z) for z in zip(df['Map_Name'], df['Total_Medals'])]
data_gold = [list(z) for z in zip(df['Map_Name'], df['Gold_Medals'])]

# ---------------------------------------------------------
# Chart 1: Predicted Total Medals
# ---------------------------------------------------------
map_total = (
    Map(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color="#FFFFFF", renderer="svg"))
    .add(
        series_name="Total Medals",  # Changed to English
        data_pair=data_total,
        maptype="world",
        is_map_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
        itemstyle_opts=opts.ItemStyleOpts(border_width=0.5, border_color="rgba(0,0,0,0.2)"),
        zoom=1.2,
        emphasis_label_opts=opts.LabelOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c}")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Predicted Total Medals for 2028 Olympics",  # Changed to English
            pos_left="center",
            title_textstyle_opts=opts.TextStyleOpts(font_size=24)
        ),
        visualmap_opts=opts.VisualMapOpts(
            max_=200,  # Max for total medals
            min_=0,
            is_piecewise=False,
            range_color=["#E0F3F8", "#4575B4", "#D73027"], # Blue to Red
            orient="horizontal",
            pos_left="center",
            pos_bottom="5%"
        ),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

# ---------------------------------------------------------
# Chart 2: Predicted Gold Medals
# ---------------------------------------------------------
map_gold = (
    Map(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color="#FFFFFF", renderer="svg"))
    .add(
        series_name="Gold Medals",  # Changed to English
        data_pair=data_gold,
        maptype="world",
        is_map_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
        itemstyle_opts=opts.ItemStyleOpts(border_width=0.5, border_color="rgba(0,0,0,0.2)"),
        zoom=1.2,
        emphasis_label_opts=opts.LabelOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c}")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Predicted Gold Medals for 2028 Olympics",  # Changed to English
            pos_left="center",
            pos_top="20px",
            title_textstyle_opts=opts.TextStyleOpts(font_size=24)
        ),
        visualmap_opts=opts.VisualMapOpts(
            max_=70,   # Max for gold medals
            min_=0,
            is_piecewise=False,
            range_color=["#FFFDE7", "#FDD835", "#F57F17"], # Gold color scale
            orient="horizontal",
            pos_left="center",
            pos_bottom="5%"
        ),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

# ---------------------------------------------------------
# Combine Charts
# ---------------------------------------------------------
page = Page(layout=Page.SimplePageLayout)
page.add(map_total)
page.add(map_gold)

output_file = "2028_Olympic_Medals_Double_Map_EN.html"
page.render(output_file)
print(f"Maps generated: {output_file}")