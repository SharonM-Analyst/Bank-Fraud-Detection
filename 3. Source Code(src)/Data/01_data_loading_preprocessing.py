# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1: Data Loading & Preprocessing
# MAGIC **Project:** Nigeria Bank Fraud Detection
# MAGIC **Purpose:** Load data from the Gold layer, perform data quality checks, and produce a clean dataset saved as a Delta table for downstream use.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Install Libraries

# COMMAND ----------
# MAGIC %pip install numpy pandas matplotlib seaborn scikit-learn plotly sweetviz

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Import Libraries

# COMMAND ----------
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

print("Libraries loaded successfully.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Data Ingestion — Load from Gold Layer (Delta Table)

# COMMAND ----------
# Read from the Gold Delta table
spark_bank_fraud = spark.read.table("workspace.mlops.gold_nibs")

# Convert to Pandas
bank_fraud = spark_bank_fraud.toPandas()

print(f"Dataset loaded: {bank_fraud.shape[0]:,} rows × {bank_fraud.shape[1]} columns")
display(bank_fraud.head(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Data Overview

# COMMAND ----------
# Shape
print(f"Rows   : {bank_fraud.shape[0]:,}")
print(f"Columns: {bank_fraud.shape[1]}")

# COMMAND ----------
# Data types, null counts, memory usage
bank_fraud.info()

# COMMAND ----------
# Statistical summary
bank_fraud.describe().T

# COMMAND ----------
# Unique values for key categorical columns
cols = ['bank', 'merchant_category', 'channel', 'location', 'age_group']
unique_vals = {col: bank_fraud[col].unique().tolist() for col in cols}
for col, vals in unique_vals.items():
    print(f"\n{col.upper()} ({len(vals)} unique): {vals}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Data Quality Checks

# COMMAND ----------
# --- Null counts per column ---
null_counts = bank_fraud.isnull().sum()
print("=== Null Counts Per Column ===")
print(null_counts[null_counts > 0])

print(f"\nTotal null values: {bank_fraud.isnull().sum().sum():,}")

# COMMAND ----------
# --- Duplicate rows ---
dups = bank_fraud.duplicated().sum()
print(f"Duplicate rows: {dups}")

# COMMAND ----------
# --- Amount sanity check ---
print("=== Amount ===")
print(f"Min : NGN {bank_fraud['amount'].min():,.2f}")
print(f"Max : NGN {bank_fraud['amount'].max():,.2f}")
neg = (bank_fraud['amount'] < 0).sum()
print(f"Negative values: {neg}  ({neg / len(bank_fraud) * 100:.2f}%)")

# COMMAND ----------
# --- Fraud class distribution ---
fraud_dist = bank_fraud['is_fraud'].value_counts()
fraud_rate = bank_fraud['is_fraud'].mean() * 100
print(f"Fraud Rate: {fraud_rate:.4f}%")
print(fraud_dist)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Data Cleaning

# COMMAND ----------
# Work on a copy — never mutate raw data
bank_fraud_clean = bank_fraud.copy()

# COMMAND ----------
# --- Convert timestamp to datetime ---
bank_fraud_clean['timestamp'] = pd.to_datetime(
    bank_fraud_clean['timestamp'], format='%Y-%m-%d %H:%M:%S'
)
print(f"Timestamp dtype: {bank_fraud_clean['timestamp'].dtype}")

# COMMAND ----------
# --- Handle nulls in fraud_technique: fill with 'Unknown' ---
if 'fraud_technique' in bank_fraud_clean.columns:
    bank_fraud_clean['fraud_technique'] = bank_fraud_clean['fraud_technique'].fillna('Unknown')
    print(f"fraud_technique nulls remaining: {bank_fraud_clean['fraud_technique'].isnull().sum()}")

# COMMAND ----------
# --- Drop columns not needed for modelling ---
cols_to_drop = [
    'fraud_technique', 'tx_count_24h', 'amount_sum_24h',
    'amount_mean_7d', 'amount_std_7d', 'tx_count_total',
    'amount_mean_total', 'amount_std_total',
    'channel_diversity', 'location_diversity',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'month_sin', 'month_cos', 'amount_rounded'
]

bank_fraud_clean = bank_fraud_clean.drop(cols_to_drop, errors='ignore')
print(f"Shape after dropping columns: {bank_fraud_clean.shape}")

# COMMAND ----------
# --- Final null check on cleaned dataset ---
print("Remaining nulls per column:")
print(bank_fraud_clean.isnull().sum()[bank_fraud_clean.isnull().sum() > 0])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Save Cleaned Dataset as Delta Table

# COMMAND ----------
# Convert back to Spark DataFrame and write as Delta table
spark_clean = spark.createDataFrame(bank_fraud_clean)

spark_clean.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.mlops.preprocessed_nibs")

print("Cleaned dataset saved to: workspace.mlops.preprocessed_nibs")
print(f"Final shape: {bank_fraud_clean.shape[0]:,} rows × {bank_fraud_clean.shape[1]} columns")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC | Step | Result |
# MAGIC |------|--------|
# MAGIC | Raw rows loaded | See above |
# MAGIC | Duplicate rows | Checked |
# MAGIC | Nulls handled | fraud_technique → 'Unknown' |
# MAGIC | Timestamp converted | datetime64 |
# MAGIC | Irrelevant columns dropped | 17 columns removed |
# MAGIC | Output table | workspace.mlops.preprocessed_nibs |

