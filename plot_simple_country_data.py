import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 0) 配置
# =========================
FILE_PATH = r"mit_timeseries_2000_2024.csv"   # MIT时序（Real_MIT_Total）
PROB_CSV  = r"mit_interpolation.csv"          # 概率表（含 Pr_Delta_Pos/Pr_Delta_Neg）
OUT_PNG   = "real_mit_final_plot.png"

TARGET_COMBOS = [
    ("Volleyball Women", "USA"),
    ("Judo", "JPN"),
    ("Cycling Track", "GBR"),
    # ("Volleyball Women", "CHN"),
]

ALL_YEARS = list(range(2000, 2025, 4))

STAR_POINTS = [
    {"Sport": "Volleyball Women", "NOC": "USA", "Year": 2008},
    {"Sport": "Volleyball Women", "NOC": "CHN", "Year": 2016},  # 若图中没画CHN线，会自动跳过
    {"Sport": "Judo",            "NOC": "JPN", "Year": 2016},
    {"Sport": "Cycling Track",   "NOC": "GBR", "Year": 2008},
]

STAR_COLOR_MODE = "match"   # "match"=跟线一致；"uniform"=统一颜色
UNIFORM_STAR_COLOR = "black"

ANNOT_PROB_COL = "Pr_Delta_Pos"
SHOW_AS_PERCENT = True


# =========================
# 文字不重叠的简单放置器
# =========================
def add_non_overlapping_annotations(fig, ax, ann_specs, pad_px=4):
    offsets = [
        (18, 14), (18, -14), (-18, 14), (-18, -14),
        (28, 0), (-28, 0), (0, 24), (0, -24),
        (40, 14), (40, -14), (-40, 14), (-40, -14),
        (60, 0), (-60, 0), (0, 40), (0, -40),
        (80, 14), (80, -14), (-80, 14), (-80, -14),
    ]

    placed_bboxes = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)

    for spec in ann_specs:
        text = spec["text"]
        xy = spec["xy"]
        color = spec.get("color", "black")

        placed = False
        for dx, dy in offsets:
            ha = "left" if dx >= 0 else "right"

            ann = ax.annotate(
                text,
                xy=xy,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=10,
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=color),
                zorder=20,
            )

            fig.canvas.draw()
            bb = ann.get_window_extent(renderer)
            bb = bb.from_extents(bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)

            inside = ax_bbox.contains(bb.x0, bb.y0) and ax_bbox.contains(bb.x1, bb.y1)
            overlap = any(bb.overlaps(prev) for prev in placed_bboxes)

            if inside and not overlap:
                placed_bboxes.append(bb)
                placed = True
                break
            else:
                ann.remove()

        if not placed:
            ax.annotate(
                text,
                xy=xy,
                xytext=(18, 14),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
                color=color,
                arrowprops=dict(arrowstyle="->", lw=1.0, color=color),
                zorder=20,
            )


# =========================
# 自动把图例放到“更空”的角落（图内）
# =========================
def choose_best_legend_loc(ax, data_xy_by_label):
    """
    简单启发式：统计每个角落附近有多少点，选点最少的角落放图例
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0, x1 = xlim
    y0, y1 = ylim
    dx = (x1 - x0)
    dy = (y1 - y0)

    corners = {
        "upper left":  (x0 + 0.15 * dx, y1 - 0.15 * dy),
        "upper right": (x1 - 0.15 * dx, y1 - 0.15 * dy),
        "lower left":  (x0 + 0.15 * dx, y0 + 0.15 * dy),
        "lower right": (x1 - 0.15 * dx, y0 + 0.15 * dy),
    }

    all_points = []
    for pts in data_xy_by_label.values():
        all_points.extend(pts)

    # 计算每个角的“邻域点数”
    best_loc = None
    best_score = None
    for loc, (cx, cy) in corners.items():
        # 邻域半径（相对坐标轴大小）
        rx = 0.35 * dx
        ry = 0.35 * dy
        score = 0
        for (px, py) in all_points:
            if abs(px - cx) <= rx and abs(py - cy) <= ry:
                score += 1
        if best_score is None or score < best_score:
            best_score = score
            best_loc = loc

    return best_loc if best_loc else "best"


def main():
    df = pd.read_csv(FILE_PATH)
    prob = pd.read_csv(PROB_CSV)

    # 骨架
    skeleton_data = []
    for sport, noc in TARGET_COMBOS:
        for year in ALL_YEARS:
            skeleton_data.append({"Sport": sport, "NOC": noc, "Year": year})
    df_skeleton = pd.DataFrame(skeleton_data)

    # 合并 & 补0
    merged = pd.merge(df_skeleton, df, on=["Sport", "NOC", "Year"], how="left")
    if "Real_MIT_Total" not in merged.columns:
        raise ValueError(
            "你的CSV里找不到列 'Real_MIT_Total'，请检查列名是否一致。"
            f" 当前列名为: {list(merged.columns)}"
        )
    merged["Real_MIT_Total"] = merged["Real_MIT_Total"].fillna(0.0)

    merged["Label"] = merged["NOC"] + " - " + merged["Sport"]

    labels = sorted(merged["Label"].unique().tolist())
    colors = sns.color_palette(n_colors=len(labels))
    palette_dict = dict(zip(labels, colors))

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(12, 7))
    ax = sns.lineplot(
        data=merged,
        x="Year",
        y="Real_MIT_Total",
        hue="Label",
        palette=palette_dict,
        marker="o",
        linewidth=2
    )

    # 坐标轴
    plt.xticks(ALL_YEARS)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(r"$M_{i,t}$", fontsize=12)  # << MIT -> Mi,t（下标）

    # 不要标题（你要求）
    # plt.title(...)

    # 画星星 + 标注
    ann_specs = []

    # 用于自动选图例位置：收集各线的点
    data_xy_by_label = {}
    for lab in labels:
        sub = merged[merged["Label"] == lab]
        data_xy_by_label[lab] = list(zip(sub["Year"].astype(float).tolist(),
                                         sub["Real_MIT_Total"].astype(float).tolist()))

    for p in STAR_POINTS:
        sel = merged[
            (merged["Sport"] == p["Sport"]) &
            (merged["NOC"] == p["NOC"]) &
            (merged["Year"] == p["Year"])
        ]
        if sel.empty:
            print(f"[WARN] 该星点不在当前图数据里（可能该线未画/组合不在TARGET_COMBOS）：{p}")
            continue

        x = float(sel["Year"].iloc[0])
        y = float(sel["Real_MIT_Total"].iloc[0])
        label = sel["Label"].iloc[0]

        if STAR_COLOR_MODE == "match":
            star_color = palette_dict.get(label, "black")
        else:
            star_color = UNIFORM_STAR_COLOR

        ax.scatter(
            x, y,
            marker="*",
            s=320,
            c=[star_color],
            edgecolors="black",
            linewidths=1.0,
            zorder=15
        )

        if ANNOT_PROB_COL not in prob.columns:
            raise ValueError(f"概率表里找不到列 {ANNOT_PROB_COL}，当前列为：{list(prob.columns)}")

        ps = prob[
            (prob["Sport"] == p["Sport"]) &
            (prob["NOC"] == p["NOC"]) &
            (prob["Year"] == p["Year"])
        ]
        if ps.empty:
            print(f"[WARN] 概率表里找不到该点：{p}，将只画星不标概率")
            continue

        val = float(ps[ANNOT_PROB_COL].iloc[0])
        if SHOW_AS_PERCENT:
            if ANNOT_PROB_COL == "Pr_Delta_Neg":
                txt = f"{p['NOC']} {p['Year']}  P($\\Delta$<0)={val*100:.1f}%"
            else:
                txt = f"{p['NOC']} {p['Year']}  P($\\Delta$>0)={val*100:.1f}%"
        else:
            txt = f"{p['NOC']} {p['Year']}  {ANNOT_PROB_COL}={val:.4f}"

        ann_specs.append({"text": txt, "xy": (x, y), "color": star_color})

    add_non_overlapping_annotations(fig, ax, ann_specs)

    # 图例放到图内空白处
    best_loc = choose_best_legend_loc(ax, data_xy_by_label)
    leg = ax.legend(loc=best_loc, title="Legend", frameon=True, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()
    print(f"绘图完成，已保存为 {OUT_PNG}")


if __name__ == "__main__":
    main()
