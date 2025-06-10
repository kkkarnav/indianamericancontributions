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
args = parser.parse_args()

os_csv = args.path
year = os_csv.split(".")[-2][-2:]

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

df = lf.collect(streaming=True)
print(df.head(10))

donors = (
    df.group_by("contrib_id")
    .agg(
        pl.col("name").first().alias("name"),
        pl.col("name_new").first().alias("name_new"),
        pl.col("orgname").first().alias("orgname"),
        pl.col("ultorg").first().alias("ultorg"),
        pl.col("gender").first().alias("gender"),
        pl.col("occupation").first().alias("occupation"),
        pl.col("employer").first().alias("employer"),
        pl.col("amount").sum().alias("total_donated"),
        pl.col("amount").count().alias("donation_count"),
        pl.col("amount").mean().alias("avg_donation"),
        pl.col("amount").median().alias("med_donation"),
    )
    .sort("total_donated", descending=True)
)

donors.write_csv(f"./data/CampaignFin{year}/donors{year}.csv")
print(donors.tail(10))

donors[:10000].write_csv(f"./data/CampaignFin{year}/top_donors{year}.csv")
print(donors.head(10))
