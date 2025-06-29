import argparse, os, time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import seaborn as sns
from collections import defaultdict
import warnings
from tqdm import tqdm
import pickle
import argparse

tqdm.pandas()
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to the indivs csv", type=str, default="./data/CampaignFin20/indivs20.txt")
parser.add_argument("-t", "--task", help="Task to run/generate", type=str, default="donors")
args = parser.parse_args()

os_csv = args.path
year = os_csv.split(".")[-2][-2:]


# for 2016
'''with open(os_csv, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
    content = content.replace('|15| ', '|15 |')  # or whatever fix you're trying
    
temp_file = "temp_fixed.csv"
with open(temp_file, 'w', encoding='utf-8') as f:
    f.write(content)'''

lf = (
        pl.scan_csv(
            os_csv,
            separator=',', 
            quote_char='|', 
            encoding='utf8-lossy', 
            has_header=False,
            new_columns=['dummy1', 'dummy2', 'contrib_id', 'name', 'recip_id', 
                        'orgname', 'ultorg', 'realcode', 'dummy3', 'amount', 
                        'street', 'city', 'state', 'zip', 'recipcode', 'type', 'dummy4', 'dummy5', 'gender', 'dummy6', 'occupation', 'employer', 'dummy7'],
            schema_overrides={'amount': pl.Float64, 'name': pl.Utf8, 'state': pl.Utf8, 'city': pl.Utf8},
            ignore_errors=True
        )
        .select(['contrib_id', 'name', 'recip_id', 'orgname', 'ultorg', 'realcode', 
                 'amount', 'street', 'city', 'state', 'zip', 'recipcode', 'type', 
                 'gender', 'occupation', 'employer'])
        # remove blank donations
        .filter(~pl.col('amount').is_null())
        # remove refunds
        .filter(pl.col('amount') > 0)
        # create a lowercase name column in the usual format
        .with_columns([
            pl.col("name").str.split(",").list.get(-1)
                .str.to_lowercase().str.strip_chars().alias("firstname"),
            pl.col("name").str.split(",").list.first()
                .str.to_lowercase().str.strip_chars().alias("lastname"),
        ])
        .with_columns([
            (pl.col("firstname") + " " + pl.col("lastname")).alias("name_new")
        ])
    )

# df = lf.collect(streaming=True)
# print(df.head(10))


if args.task == "donors":
    
    print("Aggregating donors...")
    
    donors = (
        lf.group_by("contrib_id")
        .agg(
            pl.col("name").mode().first().alias("name"),
            pl.col("name_new").mode().first().alias("name_new"),
            pl.col("orgname").mode().first().alias("orgname"),
            pl.col("ultorg").mode().first().alias("ultorg"),
            pl.col("realcode").mode().first().alias("realcode"),
            pl.col("gender").mode().first().alias("gender"),
            pl.col("occupation").mode().first().alias("occupation"),
            pl.col("employer").mode().first().alias("employer"),
            pl.col("amount").sum().alias("total_donated"),
            pl.col("amount").count().alias("donation_count"),
            pl.col("amount").mean().alias("avg_donation"),
            pl.col("amount").median().alias("med_donation"),
        )
        .sort("total_donated", descending=True)
        .collect(streaming=True)
    )

    donors.write_csv(f"./data/CampaignFin{year}/donors{year}.csv")
    print(donors.tail(10))

    donors[:10000].write_csv(f"./data/CampaignFin{year}/top_donors{year}.csv")
    print(donors.head(10))


if args.task == "donors_state":
    
    print("Aggregating donors with state info...")
    
    donors_state = (
        lf.group_by("contrib_id")
        .agg(
            pl.col("name").mode().first().alias("name"),
            pl.col("name_new").mode().first().alias("name_new"),
            pl.col("orgname").mode().first().alias("orgname"),
            pl.col("ultorg").mode().first().alias("ultorg"),
            pl.col("realcode").mode().first().alias("realcode"),
            pl.col("gender").mode().first().alias("gender"),
            pl.col("occupation").mode().first().alias("occupation"),
            pl.col("employer").mode().first().alias("employer"),
            pl.col("city").mode().first().alias("city"),
            pl.col("state").mode().first().alias("state"),
            pl.col("amount").sum().alias("total_donated"),
            pl.col("amount").count().alias("donation_count"),
            pl.col("amount").mean().alias("avg_donation"),
            pl.col("amount").median().alias("med_donation"),
        )
        .sort("total_donated", descending=True)
        .collect(streaming=True)
    )

    donors_state.write_csv(f"./data/CampaignFin{year}/donors_state{year}.csv")
    print(donors_state.tail(10))

    donors_state[:10000].write_csv(f"./data/CampaignFin{year}/top_donors_state{year}.csv")
    print(donors_state.head(10))


if args.task == "state":
    
    print("Aggregating donors...")
    
    state_stats = (
    lf.group_by("state")
    .agg(
        pl.col("amount").mean().alias("avg_amount"),
        pl.col("amount").median().alias("med_amount"),
        pl.col("amount").count().alias("total_donations"),
        pl.col("amount").filter(pl.col("ethnic") == "ind").mean().alias("avg_amount_indian"),
        pl.col("amount").filter(pl.col("ethnic") == "ind").median().alias("med_amount_indian"),
        pl.col("amount").filter(pl.col("ethnic") == "ind").count().alias("indian_donations")
    )
    .sort("avg_amount", descending=True)
    .collect(streaming=True)
)
