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
parser.add_argument("-p", "--path", help="Path to the yearly csv folder", type=str, default="./output/yearly")
parser.add_argument("-t", "--task", help="Task to run/generate", type=str, default="trunc")
args = parser.parse_args()

yearly_folder = args.path


mapping = {
    'A': 'Agribusiness',
    'B': 'Construction',
    'C': 'Tech',
    'D': 'Defense',
    'E': 'Energy',
    'F': 'Finance',
    'M': 'Misc Business',
    'H': 'Health',
    'J': 'Ideology/Single Issue',
    'K': 'Law',
    'L': 'Labor',
    'M': 'Manufacturing',
    'T': 'Transportation',
    'W': 'Other',
    'Y': 'Unknown',
    'Z': 'Party'
}

dfs = []
for path in os.listdir(yearly_folder):
    
    if path.endswith(".csv"):
        
        year = path.split("_")[1]
        print(f"Processing 20{year} df...")
        
        if args.task == "trunc":
            yearly_df = pd.read_csv(yearly_folder + "/" + path)[["contrib_id", "name_new", "orgname", "realcode", 
                                                             "gender", "occupation", "city", "state", "total_donated", 
                                                             "donation_count", "avg_donation", "med_donation", 
                                                             "combined_ratio", "indian"]]
        else:
            yearly_df = pd.read_csv(yearly_folder + "/" + path)
        
        yearly_df["cycle"] = 2000 + int(year)
        yearly_df["sector"] = yearly_df["realcode"].apply(lambda x: str(x).upper()[0]).map(mapping)
        yearly_df["name_new"] = yearly_df["name_new"].str.replace(" mr ", " ").str.replace(" dr ", " ").str.replace(" mrs ", " ").str.replace(" ms ", " ").str.lower()
        dfs.append(yearly_df)

df = pd.concat(dfs, ignore_index=True)
print(df)
print(len(df))
print(df["name_new"].value_counts())

df.to_csv("./output/donors_agg_pred_lastname_trunc.csv" 
          if args.task == "trunc" else 
          "./output/donors_agg_pred_lastname.csv", index=False)
