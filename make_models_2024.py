"""Aggregate 2024 Olympic medals by country and event (Corrected for Team Events)."""

import csv
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path("summerOly_athletes.csv")
OUTPUT_FILE = Path("medals_2024_country_event.csv")


def main() -> None:
    # 1. 使用 set 来记录 "唯一获奖事件"
    # Key 的结构: (NOC, Sport, Event, Medal)
    # 只要这个组合出现过一次，后续同队的运动员就不再重复计数
    unique_medals = set()

    with INPUT_FILE.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Year"] != "2024":
                continue

            medal = row["Medal"]
            if medal in ("No medal", "NA", ""):
                continue

            # --- 核心修正 ---
            # 这里的 Event 必须是具体的比赛小项 (e.g., "Basketball Men's Basketball")
            # 组合键：国家代码 + 项目 + 具体小项 + 奖牌颜色
            # 注意：使用 NOC 而不是 Team，因为 Team 名字可能不一致（如 "China-1", "China-2"）
            # 但如果你需要保留 Team 字段用于输出，可以保留，但去重核心最好基于 NOC。
            unique_key = (row["Team"], row["NOC"], row["Sport"], row["Event"], medal)

            unique_medals.add(unique_key)

    # 2. 聚合统计
    # 现在 unique_medals 里每一项代表国家获得的一枚实际奖牌
    # 结构: (Team, NOC, Sport, Event) -> {Gold: 0, ...}
    agg = defaultdict(lambda: {"Gold": 0, "Silver": 0, "Bronze": 0})

    for (team, noc, sport, event, medal) in unique_medals:
        base_key = (team, noc, sport, event)
        agg[base_key][medal] += 1

    # 3. 输出结果
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["Team", "NOC", "Sport", "Event", "Gold", "Silver", "Bronze", "Total"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 排序输出
        for (team, noc, sport, event), counts in sorted(agg.items()):
            total = counts["Gold"] + counts["Silver"] + counts["Bronze"]
            writer.writerow({
                "Team": team,
                "NOC": noc,
                "Sport": sport,
                "Event": event,
                "Gold": counts["Gold"],
                "Silver": counts["Silver"],
                "Bronze": counts["Bronze"],
                "Total": total,
            })

    print(f"Correctly aggregated. Written {len(agg)} unique event rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()