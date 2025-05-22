#!/usr/bin/env python3
"""indian_donor_stats.py  –  streaming‑safe, Polars‑accelerated version + SVD‑reduced logistic classifier
----------------------------------------------------------------------------

• Trains (optional) a character‑ngram + SVD + LogisticRegression model to detect Indian names.
• Streams OpenSecrets *Indivs* files via Polars lazy API (fast).
• Computes:
    - Total number of unique donors
    - Number of Indian donors
    - Percentage of Indian donors
    - Average donation amount for Indian and non‑Indian donors
    - Avg number of distinct recipients per donor
    - Top 5 states by unique Indian donors + avg $/donation
    - Top 5 states by number of Indian donations + avg $/donation
    - Top 10 cities by unique Indian donors + avg $/donation
    - Top 15 Indian donors by total amount given
    - Indian donors & avg $/donation by party
    - Top 5 recipients by % of donors who are Indian

Usage:
  python indian_donor_stats.py \
      --names_csv final_all_names_code.csv \
      --indivs20  indivs20.txt \
      --indivs22  indivs22.txt [--train_model] [--model_path path]
"""
import argparse, os, time
import numpy as np
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm
import polars as pl
from collections import defaultdict
import pickle
from ethnicseer import EthnicClassifier

# ---- Argument parsing ----
def parse_args():
    p = argparse.ArgumentParser(description="Compute Indian‑donor stats for OpenSecrets indivs files.")
    p.add_argument('--names_csv',  required=True, help='CSV of names with Country_code')
    p.add_argument('--indivs20',   required=True, help='Path to indivs20.txt')
    p.add_argument('--indivs22',   required=True, help='Path to indivs22.txt')
    p.add_argument('--train_model', action='store_true', help='Retrain name classifier')
    p.add_argument('--model_path', default='indian_name_clf.joblib', help='Load/save classifier')
    return p.parse_args()

# ---- Classifier routines ----
def train_classifier(names_csv, model_path):
    df = pd.read_csv(names_csv)
    df = df.dropna(subset=['Name'])
    X = df['Name'].astype(str)
    y = (df['Country_code'] == 'en_IN').astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    vec = CountVectorizer(analyzer='char', ngram_range=(2,4))
    svd = TruncatedSVD(n_components=100, random_state=42)
    clf = make_pipeline(vec, svd, LogisticRegression(max_iter=1000, class_weight='balanced'))

    clf.fit(Xtr, ytr)
    p, r, f, _ = precision_recall_fscore_support(yte, clf.predict(Xte), pos_label=1, average='binary')
    print(f"[Model] SVD+LogReg Prec={p:.3f} Rec={r:.3f} F1={f:.3f}\n")

    joblib.dump(clf, model_path)
    print(f"Saved classifier → {model_path}\n")
    return clf


def load_classifier(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model at {path}; rerun with --train_model")
    clf = joblib.load(path)
    print(f"Loaded classifier from {path}\n")
    return clf

# ---- Combined data processing and analysis ----
def process_file(path, clf, do_detailed_summary=False):
    """Process data using Polars with optional detailed analysis."""
    print(f"Using Polars version: {pl.__version__}")
    start = time.time()
    
     # First, scan the file to extract just unique names
    print("Extracting unique names...")
    unique_names_df = (
        pl.scan_csv(
            path,
            separator=',', 
            quote_char='|', 
            encoding='utf8-lossy', 
            has_header=False,
            new_columns=['dummy1', 'dummy2', 'contrib_id', 'name'],  # Only define up to the name column
            schema_overrides={'name': pl.Utf8},
            ignore_errors=True
        )
        .filter(
            pl.col("contrib_id").is_not_null()
            & pl.col("contrib_id").str.strip_chars().ne("")
        )
        .select(['name'])
        .with_columns(pl.col('name').str.to_uppercase().str.strip_chars())
        .unique()
        .collect()
    )
    print(f"Found {len(unique_names_df):,} unique names, classifying...")
    
    # Batch predict on unique names (much faster)
    names_list = unique_names_df['name'].drop_nulls().to_list()
    
    df = pd.read_csv("./data/wiki_name_race.csv")
    df["ethnic"] = df["race"].apply(lambda x: x.split(",")[-1])
    df["ethnic"] = df["ethnic"].apply(lambda x: "ind" if x == "IndianSubContinent" else "not")
    df["name"] = df['name_first'].str.cat(df[['name_middle', 'name_last', 'name_suffix']], sep=' ', na_rep='')
    df = df[["name", "ethnic"]]
    df = df.dropna()
    
    stats = build_name_stats(df, "name", "ethnic")
    X = []
    for name in names_list:
        features = create_features(name, stats)
        X.append(list(features.values()))
    X_new = np.array(list(X))
    
    y_probs = clf.predict_proba(X_new)[:, 1]
    y_pred_adjusted = (y_probs <= 0.01).astype(int)
    print(pd.Series(y_pred_adjusted).value_counts())
    is_indian_predictions = [bool(pred) for pred in y_pred_adjusted]
    
    pred_df = pd.DataFrame({
        "name": names_list,
        "probs": y_probs
    })
    pred_df.to_csv("./output/indiv20_wiki_probs.csv", index=False)
    
    # Create a lookup dataframe
    name_lookup = pl.DataFrame({
        'name': names_list,
        'is_indian': is_indian_predictions
    })
    name_lookup_lazy = name_lookup.lazy()  # Convert to LazyFrame for joining


    print("Name classification complete, processing data...")
    # Read the data once with all required columns for any analysis
    lf = (
        pl.scan_csv(
            path,
            separator=',', 
            quote_char='|', 
            encoding='utf8-lossy', 
            has_header=False,
            # Only define the columns we actually use
            new_columns=['dummy1', 'dummy2', 'contrib_id', 'name', 'recip_id', 
                        'dummy3', 'dummy4', 'dummy5', 'dummy6', 'amount', 
                        'dummy7', 'city', 'state', 'dummy8', 'recipcode'],
            schema_overrides={'amount': pl.Float64, 'name': pl.Utf8, 'state': pl.Utf8, 'city': pl.Utf8},
            ignore_errors=True
        )
        .select(['contrib_id', 'name', 'recip_id', 'amount', 'city', 'state', 'recipcode'])
        # Join with name_lookup to get is_indian
        .with_columns(pl.col('name').str.to_uppercase().str.strip_chars())
        .join(name_lookup_lazy, on='name', how='left')
        .filter(~pl.col('amount').is_null())
    )
    
    # Basic stats for all files (donor level)
    with tqdm(total=0, desc=f"[Polars] processing {os.path.basename(path)}", bar_format='{desc}: {elapsed}') as pbar:
        # Group by donor to get donation totals per person
        donor_df = (
            lf.select(['contrib_id', 'name', 'amount', 'is_indian'])
            .group_by('contrib_id')
            .agg([
                pl.first('name').alias('name'),
                pl.first('is_indian').alias('is_indian'),
                pl.sum('amount').alias('total')
            ])
            .collect(streaming=True)
        )

    # Calculate basic stats (these go into the summary table for all years)
    total = len(donor_df)
    nind = donor_df.filter(pl.col('is_indian')).height
    sumtot = donor_df['total'].sum()
    sumind = donor_df.filter(pl.col('is_indian'))['total'].sum()
    
    # Generate basic summary for all years
    stats = {
        'cycle': os.path.splitext(os.path.basename(path))[0],
        'unique_donors': total,
        'indian_donors': nind,
        'pct_indian': nind/total*100 if total else 0,
        'avg_$_indian': sumind/nind if nind else 0,
        'avg_$_non': (sumtot-sumind)/(total-nind) if total>nind else 0
    }
    
    elapsed = time.time() - start
    print(f"[Polars] processed {os.path.basename(path)} in {elapsed:.2f}s ({total} donors)")
    
    # If detailed summary requested (for 2020), do that using the same LazyFrame
    if do_detailed_summary:
        ind_df = lf.filter(pl.col('is_indian')).collect()
        ind = pl.LazyFrame(ind_df)
        print("\nDetailed breakdown:")
        
        # 1) avg distinct recipients per donor
        donor_recip = lf.group_by('contrib_id').agg(pl.n_unique('recip_id').alias('n_recip'))
        avg_recip_global = donor_recip.select(pl.col('n_recip').mean()).collect().item()
        avg_recip_indian = (
            donor_recip
            .join(lf.select(['contrib_id','is_indian']).unique(), on='contrib_id')
            .filter(pl.col('is_indian'))
            .select(pl.col('n_recip').mean())
            .collect().item()
        )
        print(f"Avg # of distinct recipients per donor (all): {avg_recip_global:.2f}")
        print(f"Avg # for Indian donors: {avg_recip_indian:.2f}\n")

        # 2) state-level stats
        st_stats = ind.group_by('state').agg([
        pl.n_unique('contrib_id').alias('indian_donors'),
        pl.mean('amount').alias('avg_amount'),
        pl.len().alias('n_donations')
        ])

        # Get top 5 states by unique donors
        top5_states_unique = st_stats.sort('indian_donors', descending=True).head(5).collect()
        print("Top 5 states by unique Indian donors + avg $/donation:")
        print(top5_states_unique.select(['state', 'indian_donors', 'avg_amount']))

        # Get top 5 states by number of donations
        top5_states_donations = st_stats.sort('n_donations', descending=True).head(5).collect()
        print("\nTop 5 states by number of Indian donations + avg $/donation:")
        print(top5_states_donations.select(['state', 'n_donations', 'avg_amount']))

        # 3) top 10 cities - already using ind correctly
        top10_cities = (
            ind.group_by('city')
            .agg(pl.n_unique('contrib_id').alias('indian_donors'), pl.mean('amount').alias('avg_amount'))
            .sort('indian_donors', descending=True)
            .head(10)
            .collect()
        )
        print("\nTop 10 cities by unique Indian donors + avg $/donation:")
        print(top10_cities)

        # 4) top 15 Indian donors by total amount
        top15 = (
            ind.group_by(['contrib_id','name'])
               .agg(pl.sum('amount').alias('total_given'))
               .sort('total_given', descending=True)
               .head(15)
               .collect()
        )
        print("\nTop 15 Indian donors by total given:")
        print(top15)

        # 5) party stats
        party = (
            ind.with_columns(
                pl.col('recipcode').str.slice(0, 1).alias('party')  # Extract first character
            )
            .group_by('party')
            .agg(pl.n_unique('contrib_id').alias('indian_donors'), pl.mean('amount').alias('avg_amount'))
            .filter(pl.col('party').str.len_bytes() > 0)  # Remove empty party codes
            .sort('indian_donors', descending=True)
            .collect()
        )
        print("\nIndian donors & avg $/donation by party:")
        print(party)

        # 6) top 5 recipients by % Indian donors
        recip_tot = lf.group_by('recip_id').agg(pl.n_unique('contrib_id').alias('total_donors'))
        recip_ind = ind.group_by('recip_id').agg(pl.n_unique('contrib_id').alias('indian_donors'))
        top5_recip_pct = (
            recip_tot.join(recip_ind, how='left', on='recip_id')
                     .with_columns(pl.col('indian_donors').fill_null(0))  # Handle nulls
                     .with_columns((pl.col('indian_donors')/pl.col('total_donors')*100).alias('pct_indian'))
                     .filter(pl.col('total_donors')>0)
                     .sort('pct_indian', descending=True)
                     .head(5)
                     .collect()
        )
        print("\nTop 5 recipients by % of donors who are Indian:")
        print(top5_recip_pct)
        
    return stats


def preprocess_name(name):
    parts = str(name).strip().split()
    first_name = parts[0].lower() if len(parts) > 0 else ""
    last_name = parts[-1].lower() if len(parts) > 1 else ""
    
    f4_first = first_name[:4] if len(first_name) >= 4 else first_name
    l4_first = first_name[-4:] if len(first_name) >= 4 else first_name
    f4_last = last_name[:4] if len(last_name) >= 4 else last_name
    l4_last = last_name[-4:] if len(last_name) >= 4 else last_name
    
    n_sub_names = len(parts)
    has_dash = any('-' in part for part in parts)
    
    return {
        'first_name': first_name,
        'last_name': last_name,
        'f4_first': f4_first,
        'l4_first': l4_first,
        'f4_last': f4_last,
        'l4_last': l4_last,
        'n_sub_names': min(n_sub_names, 4),
        'has_dash': int(has_dash)
    }


def build_name_stats(df, name_col='name', ethnicity_col='ethnic'):
    first_name_stats = defaultdict(lambda: defaultdict(int))
    last_name_stats = defaultdict(lambda: defaultdict(int))
    f4_first_stats = defaultdict(lambda: defaultdict(int))
    l4_first_stats = defaultdict(lambda: defaultdict(int))
    f4_last_stats = defaultdict(lambda: defaultdict(int))
    l4_last_stats = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        name_info = preprocess_name(row[name_col])
        ethnicity = row[ethnicity_col]
        
        first_name_stats[name_info['first_name']][ethnicity] += 1
        last_name_stats[name_info['last_name']][ethnicity] += 1
        f4_first_stats[name_info['f4_first']][ethnicity] += 1
        l4_first_stats[name_info['l4_first']][ethnicity] += 1
        f4_last_stats[name_info['f4_last']][ethnicity] += 1
        l4_last_stats[name_info['l4_last']][ethnicity] += 1
    
    return {
        'first_name_stats': first_name_stats,
        'last_name_stats': last_name_stats,
        'f4_first_stats': f4_first_stats,
        'l4_first_stats': l4_first_stats,
        'f4_last_stats': f4_last_stats,
        'l4_last_stats': l4_last_stats
    }


def create_features(name, stats, cats=['ind', 'not']):
    name_info = preprocess_name(name)
    features = {}
    
    for eth in cats:
        fn_counts = stats['first_name_stats'][name_info['first_name']]
        total_fn = sum(fn_counts.values())
        features[f'probability_{eth}_first_name'] = fn_counts.get(eth, 0) / (total_fn + 1)
        
        ln_counts = stats['last_name_stats'][name_info['last_name']]
        total_ln = sum(ln_counts.values())
        features[f'probability_{eth}_last_name'] = ln_counts.get(eth, 0) / (total_ln + 1)
        
        f4f_counts = stats['f4_first_stats'][name_info['f4_first']]
        total_f4f = sum(f4f_counts.values())
        features[f'probability_{eth}_first_name_f4'] = f4f_counts.get(eth, 0) / (total_f4f + 1)
        
        l4f_counts = stats['l4_first_stats'][name_info['l4_first']]
        total_l4f = sum(l4f_counts.values())
        features[f'probability_{eth}_first_name_l4'] = l4f_counts.get(eth, 0) / (total_l4f + 1)
        
        f4l_counts = stats['f4_last_stats'][name_info['f4_last']]
        total_f4l = sum(f4l_counts.values())
        features[f'probability_{eth}_last_name_f4'] = f4l_counts.get(eth, 0) / (total_f4l + 1)
        
        l4l_counts = stats['l4_last_stats'][name_info['l4_last']]
        total_l4l = sum(l4l_counts.values())
        features[f'probability_{eth}_last_name_l4'] = l4l_counts.get(eth, 0) / (total_l4l + 1)
        
        features[f'best_evidence_{eth}'] = max(
            features[f'probability_{eth}_first_name'],
            features[f'probability_{eth}_last_name']
        )
    
    features['dash_indicator'] = name_info['has_dash']
    features['n_sub_names'] = name_info['n_sub_names']
    
    return features


def is_indistinguishable(name, stats, threshold=0.15):
    
    name_info = preprocess_name(name)
    features = create_features(name, stats)
    
    cats = sorted(stats['first_name_stats'][name_info['first_name']].keys())
    psi = {cat: features[f'probability_{cat}_first_name'] for cat in cats}
    phi = {cat: features[f'probability_{cat}_last_name'] for cat in cats}
    
    indistinguishable_pairs = []
    
    for i, r1 in enumerate(cats):
        for r2 in cats[i+1:]:
            
            condition1 = (abs(psi[r1] - psi[r2]) <= threshold and 
                         abs(phi[r1] - phi[r2]) <= threshold)
            
            max_psi = max(psi.values())
            max_phi = max(phi.values())
            condition2 = (max_psi - min(psi[r1], psi[r2]) <= threshold and 
                         max_phi - min(phi[r1], phi[r2]) <= threshold)
            
            if condition1 and condition2:
                indistinguishable_pairs.append(f"{r1}-{r2}")
    
    return indistinguishable_pairs if indistinguishable_pairs else None


def handle_indistinguishables(df, stats, name_col='name'):
    
    df['indistinguishable'] = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = row[name_col]
        indistinguishable = is_indistinguishable(name, stats)
        
        if indistinguishable:
            df.at[idx, 'indistinguishable'] = ','.join(indistinguishable)
    
    return df


# ---- Main driver ----
def main():
    args = parse_args()
    if args.train_model:
        clf = train_classifier(args.names_csv, args.model_path)
    else:
        clf = load_classifier(args.model_path)

    # Process 2020 data with detailed summary
    stats20 = process_file(args.indivs20, clf, do_detailed_summary=True)
    
    # Process 2022 data (commented out for now)
    # stats22 = process_file(args.indivs22, clf, do_detailed_summary=True)
    df = pd.DataFrame([stats20]).set_index('cycle')
    
    print("\nComparison:")
    print(df.to_markdown(floatfmt='.2f'))

if __name__ == '__main__':
    main()
