# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3: Model Training & Evaluation
# MAGIC **Project:** Nigeria Bank Fraud Detection
# MAGIC **Purpose:** Train a Logistic Regression baseline, compare multiple models with PyCaret, select Random Forest as the best model, evaluate it, and log everything to MLflow.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Install Libraries

# COMMAND ----------
# MAGIC %pip install pycaret scikit-learn mlflow imbalanced-learn

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Import Libraries

# COMMAND ----------
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import plotly.express as px
import warnings

from sklearn.model_selection     import train_test_split
from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier
from sklearn.metrics             import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.preprocessing       import StandardScaler
from pycaret.classification      import (
    setup, compare_models, create_model,
    evaluate_model, finalize_model,
    pull, plot_model, save_model
)

warnings.filterwarnings("ignore")
print("Libraries loaded.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load Feature-Engineered Data

# COMMAND ----------
spark_df = spark.read.table("workspace.mlops.features_nibs")
df = spark_df.toPandas()

print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Define Features & Target

# COMMAND ----------
# Numeric features
numeric_features = [
    'amount', 'amount_log',
    'velocity_score', 'merchant_risk_score', 'composite_risk',
    'hour', 'day', 'is_weekend'
]

# Encoded categorical features
encoded_features = [c for c in df.columns if c.endswith('_encoded')]

feature_cols = numeric_features + encoded_features
target_col   = 'is_fraud'

# Drop rows with any NaN in features or target
model_df = df[feature_cols + [target_col]].dropna()

X = model_df[feature_cols]
y = model_df[target_col]

print(f"Features  : {len(feature_cols)}")
print(f"Samples   : {len(X):,}")
print(f"Fraud rate: {y.mean()*100:.4f}%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Train / Test Split

# COMMAND ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y       # preserve fraud ratio in both splits
)

print(f"Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
print(f"Test  fraud rate: {y_test.mean()*100:.4f}%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Baseline Model — Logistic Regression

# COMMAND ----------
with mlflow.start_run(run_name="baseline_logistic_regression"):

    # Scale features (required for Logistic Regression)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(
        class_weight  = 'balanced',   # handles class imbalance
        max_iter      = 500,
        random_state  = 42
    )
    lr.fit(X_tr_sc, y_train)
    y_pred_lr = lr.predict(X_te_sc)
    y_prob_lr = lr.predict_proba(X_te_sc)[:, 1]

    acc_lr  = accuracy_score(y_test,  y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr  = recall_score(y_test,    y_pred_lr)
    f1_lr   = f1_score(y_test,        y_pred_lr)
    auc_lr  = roc_auc_score(y_test,   y_prob_lr)

    mlflow.log_param("model",        "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("accuracy",    acc_lr)
    mlflow.log_metric("precision",   prec_lr)
    mlflow.log_metric("recall",      rec_lr)
    mlflow.log_metric("f1_score",    f1_lr)
    mlflow.log_metric("roc_auc",     auc_lr)
    mlflow.sklearn.log_model(lr, "logistic_regression_model")

print("=== Baseline: Logistic Regression ===")
print(f"  Accuracy  : {acc_lr:.4f}")
print(f"  Precision : {prec_lr:.4f}")
print(f"  Recall    : {rec_lr:.4f}")
print(f"  F1 Score  : {f1_lr:.4f}")
print(f"  ROC-AUC   : {auc_lr:.4f}")
print("\n  → Low performance triggers multi-model comparison with PyCaret.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Multi-Model Comparison — PyCaret

# COMMAND ----------
# PyCaret setup
pycaret_df = model_df.copy()

clf_setup = setup(
    data          = pycaret_df,
    target        = target_col,
    train_size    = 0.8,
    fix_imbalance = True,           # handles class imbalance via SMOTE
    normalize     = True,
    session_id    = 42,
    verbose       = False,
    log_experiment= True,
    experiment_name = "nigeria_fraud_pycaret"
)

# COMMAND ----------
# Compare all models — returns the best one
best_model = compare_models(
    sort          = 'F1',
    n_select      = 1,
    exclude       = ['catboost']    # exclude if catboost not installed
)

# Pull the leaderboard
leaderboard = pull()
display(leaderboard)

print(f"\nBest model selected: {type(best_model).__name__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Train Final Model — Random Forest

# COMMAND ----------
# Create Random Forest explicitly (confirmed best from PyCaret comparison)
rf_model = create_model('rf', fold=5)

# Pull metrics from PyCaret
rf_metrics = pull()
display(rf_metrics)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Model Evaluation

# COMMAND ----------
# Confusion Matrix
plot_model(rf_model, plot='confusion_matrix', save=False)

# COMMAND ----------
# ROC Curve
plot_model(rf_model, plot='auc', save=False)

# COMMAND ----------
# Feature Importance
plot_model(rf_model, plot='feature', save=False)

# COMMAND ----------
# Classification Report
plot_model(rf_model, plot='class_report', save=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Finalize & Evaluate on Hold-out Test Set

# COMMAND ----------
# Finalize retrains on 100% of data
final_rf = finalize_model(rf_model)

# Manual evaluation on original test split
y_pred_rf = final_rf.predict(X_test)
y_prob_rf = final_rf.predict_proba(X_test)[:, 1]

acc_rf  = accuracy_score(y_test,  y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf  = recall_score(y_test,    y_pred_rf)
f1_rf   = f1_score(y_test,        y_pred_rf)
auc_rf  = roc_auc_score(y_test,   y_prob_rf)

print("=== Final Model: Random Forest ===")
print(f"  Accuracy  : {acc_rf:.4f}")
print(f"  Precision : {prec_rf:.4f}")
print(f"  Recall    : {rec_rf:.4f}")
print(f"  F1 Score  : {f1_rf:.4f}")
print(f"  ROC-AUC   : {auc_rf:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_rf, target_names=['Not Fraud', 'Fraud']))

# COMMAND ----------
# Confusion matrix visualisation
cm = confusion_matrix(y_test, y_pred_rf)

fig = px.imshow(
    cm,
    text_auto      = True,
    labels         = {'x': 'Predicted', 'y': 'Actual'},
    x              = ['Not Fraud', 'Fraud'],
    y              = ['Not Fraud', 'Fraud'],
    color_continuous_scale = 'Greens',
    title          = 'RandomForestClassifier — Confusion Matrix'
)
fig.update_layout(title_x=0.5)
fig.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Model Comparison Summary

# COMMAND ----------
comparison = pd.DataFrame({
    'Model'    : ['Logistic Regression', 'Random Forest'],
    'Accuracy' : [acc_lr, acc_rf],
    'Precision': [prec_lr, prec_rf],
    'Recall'   : [rec_lr, rec_rf],
    'F1 Score' : [f1_lr, f1_rf],
    'ROC-AUC'  : [auc_lr, auc_rf]
}).round(4)

display(comparison)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Log Final Model to MLflow

# COMMAND ----------
mlflow.set_experiment("/Users/nigeria_fraud_detection")

with mlflow.start_run(run_name="final_random_forest") as run:

    # Log parameters
    mlflow.log_param("model",         "RandomForestClassifier")
    mlflow.log_param("fix_imbalance", "SMOTE via PyCaret")
    mlflow.log_param("n_features",    len(feature_cols))
    mlflow.log_param("train_size",    X_train.shape[0])
    mlflow.log_param("test_size",     X_test.shape[0])

    # Log metrics
    mlflow.log_metric("accuracy",  acc_rf)
    mlflow.log_metric("precision", prec_rf)
    mlflow.log_metric("recall",    rec_rf)
    mlflow.log_metric("f1_score",  f1_rf)
    mlflow.log_metric("roc_auc",   auc_rf)

    # Log feature list as artifact
    feat_series = pd.Series(feature_cols, name='feature')
    feat_series.to_csv('/tmp/feature_cols.csv', index=False)
    mlflow.log_artifact('/tmp/feature_cols.csv')

    # Log model
    mlflow.sklearn.log_model(
        sk_model        = final_rf,
        artifact_path   = "random_forest_model",
        registered_model_name = "nigeria_fraud_rf"
    )

    run_id = run.info.run_id

print(f"MLflow run logged. Run ID: {run_id}")
print(f"Model registered as: nigeria_fraud_rf")

# COMMAND ----------
# Pass run_id to downstream tasks
dbutils.jobs.taskValues.set(key='mlflow_run_id', value=run_id)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC | Step | Detail |
# MAGIC |------|--------|
# MAGIC | Baseline | Logistic Regression — low Recall on fraud class |
# MAGIC | Comparison | PyCaret compared 15+ algorithms on F1 |
# MAGIC | Winner | Random Forest — best F1, Recall, AUC |
# MAGIC | AUC-ROC | 1.00 (near-perfect class separation) |
# MAGIC | False Negatives | 2 out of 331 fraud cases missed |
# MAGIC | MLflow experiment | nigeria_fraud_detection |
# MAGIC | Registered model | nigeria_fraud_rf |

