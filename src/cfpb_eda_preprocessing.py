"""
cfpb_eda_preprocessing.py

Exploratory Data Analysis and Preprocessing for CFPB Complaint Data
Optimized for large files and memory efficiency.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 1. Load a sample to inspect columns
def load_sample(data_path, nrows=1000):
    sample = pd.read_csv(data_path, nrows=nrows)
    print("Sample columns:", sample.columns.tolist())
    print("Missing values:\n", sample.isnull().sum())
    print("Data types:\n", sample.dtypes)
    return sample

# 2. Load full data with filtering and chunking
def load_and_filter_data(data_path, cols, products, chunksize=100_000):
    def process_chunk(chunk):
        chunk = chunk[chunk['Product'].isin(products)]
        chunk = chunk.dropna(subset=['Consumer complaint narrative'])
        return chunk
    chunks = []
    for chunk in pd.read_csv(data_path, usecols=cols, chunksize=chunksize):
        filtered = process_chunk(chunk)
        chunks.append(filtered)
    df = pd.concat(chunks, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} complaints after filtering.")
    return df

# 3. Plot distribution of complaints by product
def plot_complaints_by_product(df, save_path=None):
    plt.figure(figsize=(10,5))
    sns.countplot(y='Product', data=df, order=df['Product'].value_counts().index)
    plt.title('Number of Complaints by Product')
    plt.xlabel('Count')
    plt.ylabel('Product')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# 4. Analyze narrative length
def analyze_narrative_length(df, save_path=None):
    df['narrative_word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10,5))
    sns.histplot(df['narrative_word_count'], bins=50, kde=True)
    plt.title('Distribution of Narrative Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Number of Complaints')
    plt.xlim(0, 500)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print('Short narratives (<=10 words):', (df['narrative_word_count'] <= 10).sum())
    print('Long narratives (>=500 words):', (df['narrative_word_count'] >= 500).sum())

# 5. Report complaints with and without narratives
def report_narrative_counts(df):
    total = len(df)
    with_narrative = df['Consumer complaint narrative'].notnull().sum()
    without_narrative = total - with_narrative
    print(f"With narrative: {with_narrative}")
    print(f"Without narrative: {without_narrative}")

# 6. Clean text for embedding
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'i am writing to file a complaint', '', text)
    text = re.sub(r'to whom it may concern', '', text)
    text = re.sub(r'dear sir or madam', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def add_cleaned_narrative(df):
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    return df

# 7. Save processed data
def save_processed_data(df, out_path):
    df.to_csv(out_path, index=False)
    print(f'Saved cleaned data to {out_path}')

if __name__ == "__main__":
    # File paths
    data_path = os.path.join('data', 'raw', 'complaints.csv')
    out_path = os.path.join('data', 'processed', 'complaints_cleaned.csv')
    # Columns of interest
    cols = [
        'Product',
        'Consumer complaint narrative',
        'Date received',
        'Company',
        'Issue',
        'Submitted via'
    ]
    products = [
        'Credit card',
        'Personal loan',
        'Buy Now, Pay Later',
        'Savings account',
        'Money transfer, virtual currency'
    ]
    # 1. Load sample
    print("--- Sample Inspection ---")
    load_sample(data_path)
    # 2. Load and filter full data
    print("--- Loading and Filtering Data ---")
    df = load_and_filter_data(data_path, cols, products)
    # 3. Plot complaints by product
    print("--- Plotting Complaints by Product ---")
    plot_complaints_by_product(df)
    # 4. Analyze narrative length
    print("--- Analyzing Narrative Length ---")
    analyze_narrative_length(df)
    # 5. Report narrative counts
    print("--- Reporting Narrative Counts ---")
    report_narrative_counts(df)
    # 6. Clean text
    print("--- Cleaning Narratives ---")
    df = add_cleaned_narrative(df)
    print(df[['Consumer complaint narrative', 'cleaned_narrative']].head())
    # 7. Save processed data
    print("--- Saving Processed Data ---")
    save_processed_data(df, out_path)
