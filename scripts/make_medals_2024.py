"""Aggregate 2024 Olympic medals by country and event from summerOly_athletes.csv."""

import csv
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path("summerOly_athletes.csv")
OUTPUT_FILE = Path("medals_2024_country_event.csv")


def main() -> None:
    agg = defaultdict(lambda: {"Gold": 0, "Silver": 0, "Bronze": 0})

    with INPUT_FILE.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Year"] != "2024":
                continue

            medal = row["Medal"]
            if medal in ("No medal", "NA", ""):
                continue

            key = (row["Team"], row["NOC"], row["Sport"], row["Event"])
            agg[key][medal] += 1

    with OUTPUT_FILE.open("w", newline="") as f:
        fieldnames = ["Team", "NOC", "Sport", "Event", "Gold", "Silver", "Bronze", "Total"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (team, noc, sport, event), counts in sorted(agg.items()):
            total = counts["Gold"] + counts["Silver"] + counts["Bronze"]
            writer.writerow(
                {
                    "Team": team,
                    "NOC": noc,
                    "Sport": sport,
                    "Event": event,
                    "Gold": counts["Gold"],
                    "Silver": counts["Silver"],
                    "Bronze": counts["Bronze"],
                    "Total": total,
                }
            )

    print(f"written {len(agg)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
