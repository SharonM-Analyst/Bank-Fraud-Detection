# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2: Feature Engineering
# MAGIC **Project:** Nigeria Bank Fraud Detection
# MAGIC **Purpose:** Load the preprocessed dataset, create date/time features, categorical features, and engineered risk signals. Save the feature-ready dataset for model training.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Import Libraries

# COMMAND ----------
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")
print("Libraries loaded.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load Preprocessed Data

# COMMAND ----------
spark_df = spark.read.table("workspace.mlops.preprocessed_nibs")
df = spark_df.toPandas()

print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
display(df.head(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Date & Time Feature Extraction

# COMMAND ----------
# Ensure timestamp is datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Extract date parts ---
df['year']        = df['timestamp'].dt.year
df['month']       = df['timestamp'].dt.month_name()
df['day_of_week'] = df['timestamp'].dt.day_name()
df['hour']        = df['timestamp'].dt.hour
df['day']         = df['timestamp'].dt.day
df['time']        = pd.to_datetime(df['timestamp']).dt.time

print("Date features created:")
print(df[['year', 'month', 'day_of_week', 'hour', 'day']].head(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Day Classification — Weekday vs Weekend

# COMMAND ----------
df['DayClassification'] = 'Weekday'
df.loc[df['day_of_week'] == 'Sunday',   'DayClassification'] = 'Weekend'
df.loc[df['day_of_week'] == 'Saturday', 'DayClassification'] = 'Weekend'

# Binary flag version (useful for models)
df['is_weekend'] = (df['DayClassification'] == 'Weekend').astype(int)

print(df['DayClassification'].value_counts())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Season Category (Nigeria-specific: Dry / Rainy)

# COMMAND ----------
df['season'] = 'Unknown'
df.loc[df['month'].isin(['November', 'December', 'January', 'February', 'March']),                      'season'] = 'Dry Season'
df.loc[df['month'].isin(['April', 'May', 'June', 'July', 'August', 'September', 'October']), 'season'] = 'Rainy Season'

print(df['season'].value_counts())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Time Bucket (Hour → Named Period)

# COMMAND ----------
df['time_bucket'] = 'Unknown'
df.loc[df['hour'].between(0,  4),  'time_bucket'] = 'Midnight (00:00–04:59)'
df.loc[df['hour'].between(5,  7),  'time_bucket'] = 'Early Morning (05:00–07:59)'
df.loc[df['hour'].between(8,  11), 'time_bucket'] = 'Morning (08:00–11:59)'
df.loc[df['hour'].between(12, 14), 'time_bucket'] = 'Midday (12:00–14:59)'
df.loc[df['hour'].between(15, 17), 'time_bucket'] = 'Afternoon (15:00–17:59)'
df.loc[df['hour'].between(18, 20), 'time_bucket'] = 'Evening (18:00–20:59)'
df.loc[df['hour'].between(21, 23), 'time_bucket'] = 'Night (21:00–23:59)'

print(df['time_bucket'].value_counts())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Month Category (Period of Month)

# COMMAND ----------
df['month_category'] = 'Unknown'
df.loc[df['day'].between(1,  10), 'month_category'] = 'Beginning (1st–10th)'
df.loc[df['day'].between(11, 20), 'month_category'] = 'Mid Month (11th–20th)'
df.loc[df['day'].between(21, 31), 'month_category'] = 'Month End (21st–31st)'

print(df['month_category'].value_counts())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Log Transform — Amount (Fix Skewness)

# COMMAND ----------
# amount_log may already exist from ELT; create if missing
if 'amount_log' not in df.columns:
    df['amount_log'] = np.log1p(df['amount'])
    print("amount_log created.")
else:
    print("amount_log already exists.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Encode Categorical Features for Modelling

# COMMAND ----------
# Identify categorical columns to encode
cat_cols = ['bank', 'merchant_category', 'channel', 'location', 'age_group',
            'DayClassification', 'season', 'time_bucket', 'month_category',
            'month', 'day_of_week']

# Label encode for tree-based models (Random Forest handles these natively)
from sklearn.preprocessing import LabelEncoder

le_dict = {}
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

print(f"Encoded {len(le_dict)} categorical columns.")
print("Encoded columns:", [c + '_encoded' for c in le_dict])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Define Final Feature Set

# COMMAND ----------
# Numeric features from the original dataset
numeric_features = [
    'amount', 'amount_log',
    'velocity_score', 'merchant_risk_score', 'composite_risk',
    'hour', 'day', 'is_weekend'
]

# Encoded categorical features
encoded_features = [col + '_encoded' for col in le_dict if col + '_encoded' in df.columns]

# All features
feature_cols = numeric_features + encoded_features
target_col   = 'is_fraud'

# Validate all features exist
missing = [f for f in feature_cols if f not in df.columns]
if missing:
    print(f"WARNING — missing features: {missing}")
else:
    print(f"Feature set ready: {len(feature_cols)} features")
    print(feature_cols)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Feature Preview & Correlation Check

# COMMAND ----------
# Distribution of key numeric features
display(df[numeric_features + [target_col]].describe().T)

# COMMAND ----------
# Correlation with target
corr_target = (
    df[feature_cols + [target_col]]
    .select_dtypes(include='number')
    .corr()[target_col]
    .sort_values(ascending=False)
    .drop(target_col)
    .reset_index()
)
corr_target.columns = ['feature', 'correlation']

fig = px.bar(
    corr_target,
    x='correlation',
    y='feature',
    orientation='h',
    text='correlation',
    title='Feature Correlation with Fraud (is_fraud)',
    labels={'correlation': 'Correlation Coefficient', 'feature': 'Feature'}
)
fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.update_layout(title_x=0.5, height=600)
fig.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Save Feature-Engineered Dataset

# COMMAND ----------
# Save full engineered dataframe
spark_feat = spark.createDataFrame(df)

spark_feat.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.mlops.features_nibs")

print("Feature dataset saved to: workspace.mlops.features_nibs")
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# COMMAND ----------
# Log feature column names for the next notebook
feature_list_str = ','.join(feature_cols)
dbutils.jobs.taskValues.set(key='feature_cols', value=feature_list_str)
dbutils.jobs.taskValues.set(key='target_col',   value=target_col)

print(f"\nPassed to next task:")
print(f"  feature_cols : {feature_list_str}")
print(f"  target_col   : {target_col}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC | Feature Group | Count |
# MAGIC |---------------|-------|
# MAGIC | Date/time features | year, month, day, hour, day_of_week |
# MAGIC | Day classification | DayClassification, is_weekend |
# MAGIC | Season | Dry Season / Rainy Season |
# MAGIC | Time buckets | 7 periods |
# MAGIC | Month category | Beginning / Mid / Month End |
# MAGIC | Log transform | amount_log |
# MAGIC | Encoded categoricals | bank, channel, location, merchant_category, age_group + time features |
# MAGIC | Output table | workspace.mlops.features_nibs |

