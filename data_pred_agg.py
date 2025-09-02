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

occupation_mapping = {
    'PHYSICIAN': 'Health',
    'PHARMACIST': 'Health',
    'NURSE': 'Health',
    'DOCTOR': 'Health',
    'MEDICAL DOCTOR': 'Health',
    'DENTIST': 'Health',
    'PATHOLOGIST': 'Health',
    'OPHTHALMOLOGIST': 'Health',
    'OPTOMETRIST': 'Health',
    'ANESTHESIOLOGIST': 'Health',
    'VETERINARIAN': 'Health',
    'PSYCHOLOGIST': 'Health',
    'PSYCHIATRIST': 'Health',
    'PSYCHOTHERAPIST': 'Health',
    'THERAPIST': 'Health',
    'COUNSELOR': 'Health',
    'CHIROPRACTOR': 'Health',
    'EMERGENCY PHYSICIAN': 'Health',
    'ORTHODONTIST': 'Health',
    'CRNA': 'Health',
    'RN': 'Health',
    'REGISTERED NURSE': 'Health',
    'NURSE PRACTITIONER': 'Health',
    'SURGEON': 'Health',
    'ORTHOPAEDIC SURGEON': 'Health',
    'ORAL SURGEON': 'Health',
    'DOCTOR OF OPTOMETRY': 'Health',
    'PHYSICAL THERAPIST': 'Health',
    'HEALTHCARE': 'Health',
    'DIAGNOSTIC RADIOLOGIST': 'Health',
    'CFO': 'Finance',
    'CHIEF FINANCIAL OFFICER': 'Finance',
    'CONSULTING': 'Finance',
    'MANAGEMENT CONSULTANT': 'Finance',
    'ACCOUNTANT': 'Finance',
    'CPA': 'Finance',
    'BOOKKEEPER': 'Finance',
    'FINANCE': 'Finance',
    'INVESTOR': 'Finance',
    'BANKER': 'Finance',
    'BANKING': 'Finance',
    'INVESTMENT BANKER': 'Finance',
    'INVESTMENT ADVISOR': 'Finance',
    'BUSINESS': 'Finance',
    'BUSINESS MANAGER': 'Finance',
    'FINANCIAL ADVISOR': 'Finance',
    'INVESTMENT': 'Finance',
    'INVESTMENTS': 'Finance',
    'BROKER': 'Finance',
    'ECONOMIST': 'Finance',
    'INSURANCE': 'Finance',
    'INSURANCE AGENT': 'Finance',
    'REALTOR': 'Finance',
    'REAL ESTATE': 'Finance',
    'REAL ESTATE BROKER': 'Finance',
    'REAL ESTATE DEVELOPER': 'Finance',
    'REAL ESTATE AGENT': 'Finance',
    'REAL ESTATE INVESTOR': 'Finance',
    'ATTORNEY': 'Law',
    'LAWYER': 'Law',
    'LOBBYIST': 'Law',
    'PARALEGAL': 'Law',
    'PRODUCT MANAGER': 'Tech',
    'COMPUTER PROGRAMMER': 'Tech',
    'DEVELOPER': 'Tech',
    'SOFTWARE ENGINEER': 'Tech',
    'IT': 'Tech',
    'SOFTWARE DEVELOPER': 'Tech',
    'SOFTWARE': 'Tech',
    'SCIENTIST': 'Tech',
    'PROGRAMMER': 'Tech',
    'FARMER': 'Agribusiness',
    'RANCHER': 'Agribusiness',
    'DRIVER': 'Transportation',
    'TRUCK DRIVER': 'Transportation',
    'PILOT': 'Transportation',
    'AIRLINE PILOT': 'Transportation',
    'AIR TRAFFIC CONTROLLER': 'Transportation',
    'PROPERTY MANAGER': 'Construction',
    'CONSTRUCTION': 'Construction',
    'ELECTRICIAN': 'Construction',
    'ELECTRICAL ENGINEER': 'Construction',
    'ARCHITECT': 'Construction',
    'LABORER': 'Labor',
    'FACTORY WORKER': 'Labor',
    'PROFESSOR': 'Education',
    'TEACHER': 'Education',
    'CLASSROOM TEACHER': 'Education',
    'EDUCATION': 'Education',
    'EDUCATOR': 'Education',
    'INSTRUCTOR': 'Education',
    'LIBRARIAN': 'Education',
    'WRITER': 'Media',
    'MUSICIAN': 'Media',
    'PUBLISHER': 'Media',
    'EDITOR': 'Media',
    'ACTOR': 'Media',
    'AUTHOR': 'Media',
    'FILMMAKER': 'Media',
    'ACTRESS': 'Media',
    'JOURNALIST': 'Media',
    'FILM PRODUCER': 'Media',
    'TV PRODUCER': 'Media',
    'PUBLISHING': 'Media',
    'PHOTOGRAPHER': 'Media',
    'FREELANCE WRITER': 'Media',
    'WRITER/EDITOR': 'Media',
    'LITERARY AGENT': 'Media',
    'COMPOSER': 'Media',
    'ENTERTAINER': 'Media',
    'WRITER/PRODUCER': 'Media',
    'SCREENWRITER': 'Media',
    'PUBLIC RELATIONS': 'Media',
    'ADVERTISING': 'Media',
    'SONGWRITER': 'Media',
    'MEDIA': 'Media',
    'ENTERTAINMENT': 'Media',
    'FILM DIRECTOR': 'Media',
    'TALENT AGENT': 'Media',
    'TELEVISION PRODUCER': 'Media',
    'MARKETING DIRECTOR': 'Media',
    'MARKETING MANAGER': 'Media',
    'VIDEO PRODUCER': 'Media',
    'SINGER': 'Media',
    'FILM MAKER': 'Media',
    'COMMUNICATIONS': 'Media',
    'TELECOMMUNICATIONS': 'Media',
    'CREATIVE DIRECTOR': 'Media',
    'FILM EDITOR': 'Media',
    'BROADCASTER': 'Media',
    'MUSIC PUBLISHER': 'Media',
    'NOVELIST': 'Media',
    'VIDEO EDITOR': 'Media',
    'ARTIST': 'Media',
    'MUSIC PRODUCER': 'Media',
    'GRAPHIC DESIGNER': 'Media',
    'DESIGNER': 'Media',
    'TV EXECUTIVE': 'Media',
    'ART DIRECTOR': 'Media',
    'VIDEOGRAPHER': 'Media',
    'AUDIO ENGINEER': 'Media',
    'CINEMATOGRAPHER': 'Media',
    'MUSIC': 'Media',
    'TV WRITER': 'Media',
    'MEDIA EXECUTIVE': 'Media',
    'FILM PRODUCTION': 'Media',
    'FILM': 'Media',
    'COMEDIAN': 'Media',
    'PLAYWRIGHT': 'Media',
    'MARKETING EXECUTIVE': 'Media',
    'PERFORMER': 'Media',
    'FREELANCE EDITOR': 'Media',
    'ADVERTISING SALES': 'Media',
    'EDITOR/WRITER': 'Media',
    'CASTING DIRECTOR': 'Media',
    'BOOK PUBLISHER': 'Media',
    'COPY EDITOR': 'Media',
    'VIDEO PRODUCTION': 'Media',
    'ANIMATOR': 'Media',
    'OPERA SINGER': 'Media',
    'TELEVISION EXECUTIVE': 'Media',
    'TELECOM': 'Media',
    'TV PRODUCTION': 'Media',
    'BROADCASTING': 'Media',
    'DOCUMENTARY FILMMAKER': 'Media',
    'MARKETING': 'Media',
    'PRODUCER': 'Media',
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
        yearly_df["sector"] = yearly_df.apply(
            lambda row: occupation_mapping.get(str(row["occupation"]).upper(), row["sector"])
            if str(row["sector"]) not in ["Party", "Ideology/Single Issue"] and str(row["occupation"]).upper() in occupation_mapping
            else row["sector"],
            axis=1
        )
        
        yearly_df["name_new"] = yearly_df["name_new"].str.replace(" mr ", " ").str.replace(" dr ", " ").str.replace(" mrs ", " ").str.replace(" ms ", " ").str.lower()
        dfs.append(yearly_df)

df = pd.concat(dfs, ignore_index=True)
print(df)
print(len(df))
print(df["name_new"].value_counts())

df.to_csv("./output/donors_agg_pred_lastname_trunc.csv" 
          if args.task == "trunc" else 
          "./output/donors_agg_pred_lastname.csv", index=False)
