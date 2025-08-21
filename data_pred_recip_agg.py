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
parser.add_argument("-p", "--path", help="Path to the compiled yearly dfs csv", type=str, default="./output/donors_agg_pred_lastname.csv")
parser.add_argument("-t", "--task", help="Task to run/generate", type=str, default="trunc")
args = parser.parse_args()


pac_party_mapping = {
    "Stop Republicans PAC": "D",
    "Senate Majority PAC": "D",
    "314 Action Fund": "D",
    "Progressive Takeover": "D",
    "Black Americans to Re-Elect the President": "R",
    "National Victory Action Fund": "R",
    "MeidasTouch": "D",
    "Swing Left": "D",
    "Democratic Strategy Institute": "D",
    "Senate Leadership Fund": "R",
    "America First Action": "R",
    "Future Forward USA": "D",
    "Unite the Country": "D",
    "Need to Impeach": "D",
    "Tech for Campaigns": "D",
    "Georgia Honor": "D",
    "The Georgia Way": "D",
    "Citizens for Free Enterprise": "R",
    "Tom Steyer PAC": "D",
    "GOPAC": "R",
    "LMG PAC": "D",
    "Democratic Majority for Israel": "D",
    "Plains PAC": "R",
    "Future Now Fund PAC": "D",
    "House Freedom Action": "R",
    "Way to Lead PAC": "D",
    "Better Future Michigan Fund": "R",
    "Digidems PAC": "D",
    "Justice & Public Safety": "D",
    "Casa in Action PAC": "D",
    "Conservative Outsider PAC": "R",
    "Save America Fund": "D",
    "People Standing Strong": "D",
    "State Government Citizens' Campaign": "D",
    "Mind the Gap": "D",
    "Elect Democratic Women": "D",
    "Everyday People PAC": "D",
    "For Our Families PAC": "D",
    "Save the US Senate PAC": "R",
    "One Vote at a Time": "D",
    "Humanity Forward Fund": "D",
    "American Patriots PAC": "R",
    "Virginia Plus PAC": "D",
    "Valor America": "R",
    "United We Win": "D",
    "New South Super PAC": "D",
    "March On PAC": "D",
    "L PAC": "D",
    "Louisiana Legacy PAC": "R",
    "Our Future United": "D",
    "New American Jobs Fund": "D",
    "Patriots of America PAC": "R",
    "Sister District Project": "D",
    "Abolitionists PAC": "D",
    "California Democracy Ventures Fund": "D",
    "WinRed": "R"
}


def read_csvs_to_merge(year):

    cands_csv = f"./data/CampaignFin{year}/cands{year}.txt"
    cmtes_csv = f"./data/CampaignFin{year}/cmtes{year}.txt"

    cands_lf = (
            pl.scan_csv(
                cands_csv,
                separator=',', 
                quote_char='|', 
                encoding='utf8-lossy', 
                has_header=False,
                new_columns=['dummy1', 'id', 'recip_id', 'name', 'party', 
                            'seat', 'seat_current', 'ran_general', 'ran_ever', 'type', 
                            'recipcode', 'nopacs'],
                ignore_errors=True
            )
            .select(['id', 'recip_id', 'name', 'party', 'seat', 'seat_current', 
                    'ran_general', 'ran_ever', 'type', 'recipcode'])
        )

    cands = cands_lf.collect()
    cands = cands.to_pandas()
    cands = cands.drop_duplicates(subset=['recip_id', 'name', 'party', 'seat'], keep='last')
    
    cmtes_lf = (
            pl.scan_csv(
                cmtes_csv,
                separator=',', 
                quote_char='|', 
                encoding='utf8-lossy', 
                has_header=False,
                new_columns=['dummy1', 'cmte_id', 'pac_short', 'affiliate', 'pac', 
                            'recip_id', 'recipcode', 'cand_id', 'party', 'prim_code', 
                            'source', 'sensitive', 'foreign', 'active'],
                ignore_errors=True
            )
            .select(['cmte_id', 'pac_short', 'affiliate', 'pac',
                    'recip_id', 'recipcode', 'cand_id', 'party', 'prim_code',
                    'source', 'sensitive', 'foreign', 'active'])
        )
    cmtes = cmtes_lf.collect(streaming=True)
    cmtes = cmtes.to_pandas()
    
    return cands, cmtes


def calculate_dem_ratio(df):
    
    party_totals = (
        df.groupby(['contrib_id', 'name_new', 'party'])['total_donated']
        .sum()
        .unstack()
        .fillna(0)
    )
    
    try:
        party_totals['dem_ratio'] = (party_totals.get('D', 0) / (party_totals.get('D', 0) + party_totals.get('R', 0)))
    except ZeroDivisionError:
        party_totals['dem_ratio'] = (party_totals.get('D', 0))
    party_totals['dem_ratio'] = 2 * party_totals['dem_ratio'] - 1

    df = df.merge(party_totals['dem_ratio'].reset_index(),on=['contrib_id', 'name_new'],how='left')
    return df


donors_csv = args.path
df = pd.read_csv(donors_csv)

recip_df = pd.DataFrame()
for year in tqdm(["00", "02", "04", "06", "08", 10, 12, 14, 16, 18, 20, 22]):

    yearly_df = df[df["cycle"] == 2000 + int(year)].copy()
    yearly_recip_csv = f"./data/CampaignFin{year}/donors_recip{year}.csv"
    yearly_recip_df = pd.read_csv(yearly_recip_csv)
    
    cands, cmtes = read_csvs_to_merge(year)
    
    yearly_recip_df = yearly_recip_df.merge(yearly_df[['contrib_id', 'sector', 'cycle', 'indian', 'combined_ratio']], on='contrib_id', how='left')
    yearly_recip_df = yearly_recip_df.merge(cands, on='recip_id', how='left')
    yearly_recip_df = yearly_recip_df.merge(cmtes[
        ["cmte_id", "pac_short", "affiliate", "pac", "recip_id", "recipcode", "cand_id", "party"]], 
        left_on='recip_id', right_on="cmte_id", how='left', suffixes=[None, "_pac"]
        ).drop_duplicates(subset=['contrib_id', 'recip_id', 'cmte_id', 'cycle'], keep='last')

    yearly_recip_df.loc[yearly_recip_df["pac"].notna(), "name_y"] = yearly_recip_df.loc[yearly_recip_df["pac"].notna(), "pac"]
    yearly_recip_df.loc[yearly_recip_df["pac_short"].notna(), "name_y"] = yearly_recip_df.loc[yearly_recip_df["pac_short"].notna(), "pac_short"]
    yearly_recip_df.loc[yearly_recip_df["pac_short"].notna(), "party"] = yearly_recip_df.loc[yearly_recip_df["pac_short"].notna(), "party_pac"]
    
    yearly_recip_df["recip_is_pac"] = False
    yearly_recip_df.loc[yearly_recip_df["pac_short"].notna(), "recip_is_pac"] = True
    
    yearly_recip_df.loc[yearly_recip_df["recip_is_pac"] == True, "party"] = yearly_recip_df.loc[yearly_recip_df["recip_is_pac"] == True, "name_y"].map(pac_party_mapping)
    yearly_recip_df = calculate_dem_ratio(yearly_recip_df)
    yearly_recip_df["level"] = yearly_recip_df.apply(lambda x: "Senate" if "S1" == str(x["seat"]) or "S2" == str(x["seat"]) else 
                                           "President" if str(x["seat"]) == "PRES" else 
                                           "House" if type(x["seat"]) == str and len(str(x["seat"])) == 4 else 
                                           "PAC" if x["recip_is_pac"] else 
                                           "Unknown", axis=1)
    
    if args.task == "trunc":
        yearly_recip_df = yearly_recip_df.loc[:, ["contrib_id", "recip_id", "name_new", "realcode", "gender", 
                             "occupation", "city", "state", "total_donated", "donation_count",
                             "avg_donation", "med_donation", "sector", "cycle", "indian", "combined_ratio",
                             "name_y", "party", "seat", "ran_general", "type", "cmte_id", "pac_short", 
                             "recip_id_pac", "cand_id", "recip_is_pac", "dem_ratio", "level"]]
    
    recip_df = pd.concat([recip_df, yearly_recip_df], axis=0)
    

print(recip_df)
print(len(recip_df))
print(recip_df["name_new"].value_counts())
print(recip_df["name_y"].value_counts())
print(recip_df.columns)

recip_df.to_csv("./output/donors_recip_agg_pred_lastname_trunc.csv" 
          if args.task == "trunc" else 
          "./output/donors_recip_agg_pred_lastname.csv", index=False)
