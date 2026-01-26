import pandas as pd

try:
    hosts = pd.read_csv("summerOly_hosts.csv", encoding='ISO-8859-1')
    print("Columns:", hosts.columns.tolist())
    print("Columns repr:", [repr(c) for c in hosts.columns])
    print(hosts.head())
except Exception as e:
    print(e)
