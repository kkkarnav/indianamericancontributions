import argparse, os, time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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


def read_name_dataset_csv(csv_path, indian):
    
    print(f"Reading name dataset from {csv_path}...")

    # https://github.com/philipperemy/name-dataset
    df = pd.read_csv(csv_path)
    df.columns = ['firstname', 'lastname', 'gender', 'ethnicity']
    
    df['firstname'] = df['firstname'].apply(lambda x: x.split(" ")[0].strip() if " " in str(x) else str(x).strip())
    df['lastname'] = df['lastname'].apply(lambda x: x.split(" ")[-1].strip() if " " in str(x) else str(x).strip())
    df['name'] = df['firstname'].apply(lambda x: x.lower()) + ' ' + df['lastname'].apply(lambda x: x.lower())
    df["indian"] = df["ethnicity"].apply(lambda x: indian)

    df = df[
        (df['firstname'].str.match(r'^[A-Za-z]+$', na=False)) & 
        (df['firstname'].str.len() > 1) &
        (df['firstname'].str.lower() != 'nan') &
        (df['lastname'].str.match(r'^[A-Za-z]+$', na=False)) &
        (df['lastname'].str.len() > 1) &
        (df['lastname'].str.lower() != 'nan')
    ]

    return df[['firstname', 'lastname', 'name', 'indian']]


def calculate_name_rates(df):
    
    print("Calculating name rates...")

    firstname_counts = df['firstname'].value_counts().reset_index().rename(columns={'count': 'firstname_count'})
    firstname_counts['firstname_rate'] = (firstname_counts['firstname_count'] / len(df)) * 100
    df = df.merge(firstname_counts[['firstname', 'firstname_count', 'firstname_rate']], on='firstname', how='left')
    
    lastname_counts = df['lastname'].value_counts().reset_index().rename(columns={'count': 'lastname_count'})
    lastname_counts['lastname_rate'] = (lastname_counts['lastname_count'] / len(df)) * 100
    df = df.merge(lastname_counts[['lastname', 'lastname_count', 'lastname_rate']], on='lastname', how='left')
    
    return df


def calculate_name_ratios(df1, df2, chunk_size=100000):
    
    print("Calculating name ratios...")
    
    all_firstnames = pd.concat([
        df1['firstname'].drop_duplicates(),
        df2['firstname'].drop_duplicates()
    ]).drop_duplicates()
    all_lastnames = pd.concat([
        df1['lastname'].drop_duplicates(),
        df2['lastname'].drop_duplicates()
    ]).drop_duplicates()
    firstname_dfs, lastname_dfs = [], []

    for i in tqdm(range(0, len(all_lastnames), chunk_size)):
        
        chunk_firstnames = all_firstnames.iloc[i:i + chunk_size]
        chunk_lastnames = all_lastnames.iloc[i:i + chunk_size]
        
        chunk_first1 = df1[df1['firstname'].isin(chunk_firstnames)]
        rates_first1 = chunk_first1[['firstname', 'firstname_count', 'firstname_rate']].drop_duplicates()
        chunk_last1 = df1[df1['lastname'].isin(chunk_lastnames)]
        rates_last1 = chunk_last1[['lastname', 'lastname_count', 'lastname_rate']].drop_duplicates()
        
        chunk_first2 = df2[df2['firstname'].isin(chunk_firstnames)]
        rates_first2 = chunk_first2[['firstname', 'firstname_count', 'firstname_rate']].drop_duplicates()
        chunk_last2 = df2[df2['lastname'].isin(chunk_lastnames)]
        rates_last2 = chunk_last2[['lastname', 'lastname_count', 'lastname_rate']].drop_duplicates()
        
        merged_first = pd.merge(rates_first1, rates_first2, on='firstname', how='outer', suffixes=('_us', '_india')).fillna(0)
        merged_last = pd.merge(rates_last1, rates_last2, on='lastname', how='outer', suffixes=('_us', '_india')).fillna(0)
        
        merged_first['ratio'] = (merged_first['firstname_rate_india'] / merged_first['firstname_rate_us']).replace(float('inf'), 1000)
        merged_last['ratio'] = (merged_last['lastname_rate_india'] / merged_last['lastname_rate_us']).replace(float('inf'), 1000)
        
        firstname_dfs.append(merged_first)
        lastname_dfs.append(merged_last)
        
    firstname_ratios = pd.concat(firstname_dfs)
    lastname_ratios = pd.concat(lastname_dfs)
    
    return firstname_ratios, lastname_ratios


us_csv = "../data/US.csv"
in_csv = "../data/IN.csv"

df_us = calculate_name_rates(read_name_dataset_csv(us_csv, indian=False))
df_in = calculate_name_rates(read_name_dataset_csv(in_csv, indian=True))

firstname_df, lastname_df = calculate_name_ratios(df_us, df_in)
firstname_df.to_csv("../output/USIN_firstnames_ratios.csv", index=False)
lastname_df.to_csv("../output/USIN_lastnames_ratios.csv", index=False)
