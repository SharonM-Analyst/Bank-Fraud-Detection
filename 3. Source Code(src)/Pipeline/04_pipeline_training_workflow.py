# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 4: Pipeline Training Workflow
# MAGIC **Project:** Nigeria Bank Fraud Detection
# MAGIC **Purpose:** Orchestration controller notebook. Runs the full pipeline — Preprocessing → Feature Engineering → Model Training — end-to-end. Triggered automatically when a new file lands in the S3 bucket.
# MAGIC
# MAGIC > **This notebook is designed to be called by a Databricks Job.**
# MAGIC > Each step calls `dbutils.notebook.run()` to execute the upstream notebooks in sequence.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pipeline Architecture
# MAGIC ```
# MAGIC S3 Bucket (new file arrives)
# MAGIC        │
# MAGIC        ▼
# MAGIC  Databricks Job Trigger
# MAGIC        │
# MAGIC        ▼
# MAGIC  [Task 1] ELT Pipeline        →  workspace.mlops.gold_nibs
# MAGIC        │
# MAGIC        ▼
# MAGIC  [Task 2] Preprocessing        →  workspace.mlops.preprocessed_nibs
# MAGIC        │
# MAGIC        ▼
# MAGIC  [Task 3] Feature Engineering  →  workspace.mlops.features_nibs
# MAGIC        │
# MAGIC        ▼
# MAGIC  [Task 4] Model Training       →  MLflow: nigeria_fraud_rf
# MAGIC        │
# MAGIC        ▼
# MAGIC  [Task 5] Pipeline Logging     →  Run summary + Slack/email alert
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Import Utilities

# COMMAND ----------
import mlflow
import datetime
import json

print(f"Pipeline started at: {datetime.datetime.now()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Pipeline Configuration

# COMMAND ----------
# ---- CONFIGURE THESE PATHS TO MATCH YOUR ENVIRONMENT ----
NOTEBOOK_BASE_PATH = "/Workspace/Users/your_email/notebooks"

NOTEBOOKS = {
    "preprocessing"       : f"{NOTEBOOK_BASE_PATH}/01_data_loading_preprocessing",
    "feature_engineering" : f"{NOTEBOOK_BASE_PATH}/02_feature_engineering",
    "model_training"      : f"{NOTEBOOK_BASE_PATH}/03_model_training_evaluation",
}

TIMEOUT_SECONDS = 3600   # 1 hour per notebook
PIPELINE_NAME   = "nigeria_fraud_training_pipeline"

# Source S3 path (for logging purposes — trigger is configured in the Job UI)
S3_SOURCE_PATH = "s3://your-bucket/fraud-data/incoming/"

print(f"Pipeline: {PIPELINE_NAME}")
print(f"Source  : {S3_SOURCE_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Pipeline Run Logger

# COMMAND ----------
pipeline_log = {
    "pipeline_name": PIPELINE_NAME,
    "start_time"   : str(datetime.datetime.now()),
    "steps"        : {}
}

def log_step(step_name, status, duration_s=None, error=None):
    pipeline_log["steps"][step_name] = {
        "status"    : status,
        "duration_s": duration_s,
        "error"     : str(error) if error else None,
        "timestamp" : str(datetime.datetime.now())
    }
    icon = "✅" if status == "SUCCESS" else "❌"
    print(f"  {icon}  {step_name}: {status}" + (f" ({duration_s:.1f}s)" if duration_s else ""))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Step 1 — Data Preprocessing

# COMMAND ----------
print("=" * 60)
print("STEP 1: Data Loading & Preprocessing")
print("=" * 60)

step_start = datetime.datetime.now()
try:
    result = dbutils.notebook.run(
        NOTEBOOKS["preprocessing"],
        timeout_seconds = TIMEOUT_SECONDS,
        arguments       = {}
    )
    duration = (datetime.datetime.now() - step_start).total_seconds()
    log_step("preprocessing", "SUCCESS", duration)
    print(f"  Output: {result}")

except Exception as e:
    duration = (datetime.datetime.now() - step_start).total_seconds()
    log_step("preprocessing", "FAILED", duration, e)
    print(f"  ERROR: {e}")
    raise Exception(f"Pipeline aborted at preprocessing step: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Step 2 — Feature Engineering

# COMMAND ----------
print("=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

step_start = datetime.datetime.now()
try:
    result = dbutils.notebook.run(
        NOTEBOOKS["feature_engineering"],
        timeout_seconds = TIMEOUT_SECONDS,
        arguments       = {}
    )
    duration = (datetime.datetime.now() - step_start).total_seconds()
    log_step("feature_engineering", "SUCCESS", duration)
    print(f"  Output: {result}")

except Exception as e:
    duration = (datetime.datetime.now() - step_start).total_seconds()
    log_step("feature_engineering", "FAILED", duration, e)
    print(f"  ERROR: {e}")
    raise Exception(f"Pipeline aborted at feature engineering step: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Step 3 — Model Training & Evaluation

# COMMAND ----------
print("=" * 60)
print("STEP 3: Model Training & Evaluation")
print("=" * 60)

step_start = datetime.datetime.now()
try:
    result = dbutils.notebook.run(
        NOTEBOOKS["model_training"],
        timeout_seconds = TIMEOUT_SECONDS,
        arguments       = {}
    )
    duration = (datetime.datetime.now() - step_start).total_seconds()
    log_step("model_training", "SUCCESS", duration)
    print(f"  Output: {result}")

except Exception as e:
    duration = (datetime.datetime.now() - step_start).total_seconds()
    log_step("model_training", "FAILED", duration, e)
    print(f"  ERROR: {e}")
    raise Exception(f"Pipeline aborted at model training step: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Pipeline Summary & MLflow Logging

# COMMAND ----------
pipeline_log["end_time"]   = str(datetime.datetime.now())
pipeline_log["status"]     = "SUCCESS"

start = datetime.datetime.fromisoformat(pipeline_log["start_time"])
end   = datetime.datetime.fromisoformat(pipeline_log["end_time"])
total_duration = (end - start).total_seconds()
pipeline_log["total_duration_s"] = total_duration

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Status   : ✅ SUCCESS")
print(f"  Duration : {total_duration/60:.1f} minutes")

print("\n  Step Breakdown:")
for step, info in pipeline_log["steps"].items():
    d = info.get('duration_s', 0)
    print(f"    {step:<25} {info['status']:<10} {d/60:.1f} min")

# COMMAND ----------
# Log pipeline run to MLflow
mlflow.set_experiment("/Users/nigeria_fraud_detection")

with mlflow.start_run(run_name=f"pipeline_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"):
    mlflow.log_param("pipeline_name",   PIPELINE_NAME)
    mlflow.log_param("trigger_source",  S3_SOURCE_PATH)
    mlflow.log_metric("total_duration_s", total_duration)
    mlflow.log_metric("steps_completed",  len(pipeline_log["steps"]))

    # Save full log as artifact
    log_path = '/tmp/pipeline_log.json'
    with open(log_path, 'w') as f:
        json.dump(pipeline_log, f, indent=2)
    mlflow.log_artifact(log_path)

print("\nPipeline run logged to MLflow.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. (Optional) Data Drift Check — Pre-training Gate

# COMMAND ----------
# MAGIC %md
# MAGIC > **Next step:** Uncomment and configure this block once monitoring is in place.
# MAGIC > This gate will compare incoming data statistics against the training baseline
# MAGIC > and abort if data drift exceeds a threshold.

# COMMAND ----------
# import json
#
# def check_data_drift(new_df, baseline_stats_path):
#     """
#     Compare mean/std of key features against baseline.
#     Raises an alert if drift > threshold.
#     """
#     with open(baseline_stats_path) as f:
#         baseline = json.load(f)
#
#     drift_report = {}
#     threshold    = 0.2   # 20% relative change triggers alert
#
#     for col in ['amount', 'velocity_score', 'composite_risk']:
#         if col in new_df.columns and col in baseline:
#             new_mean  = new_df[col].mean()
#             base_mean = baseline[col]['mean']
#             drift     = abs(new_mean - base_mean) / (base_mean + 1e-8)
#             drift_report[col] = {'drift': drift, 'alert': drift > threshold}
#             if drift > threshold:
#                 print(f"⚠️  DRIFT ALERT — {col}: {drift:.2%} change from baseline")
#
#     return drift_report

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Databricks Job Configuration Reference
# MAGIC
# MAGIC Configure this pipeline as a **Databricks Job** with the following settings:
# MAGIC
# MAGIC ### Trigger
# MAGIC - **Type:** File Arrival
# MAGIC - **S3 path:** `s3://your-bucket/fraud-data/incoming/`
# MAGIC - **File pattern:** `*.csv` or `*.parquet`
# MAGIC
# MAGIC ### Tasks (in order)
# MAGIC | Task Name | Notebook | Depends On |
# MAGIC |-----------|----------|------------|
# MAGIC | elt_pipeline | ELT pipeline notebook | — |
# MAGIC | preprocessing | 01_data_loading_preprocessing | elt_pipeline |
# MAGIC | feature_engineering | 02_feature_engineering | preprocessing |
# MAGIC | model_training | 03_model_training_evaluation | feature_engineering |
# MAGIC | pipeline_workflow | 04_pipeline_training_workflow | model_training |
# MAGIC
# MAGIC ### Cluster
# MAGIC - **Type:** Job Cluster (auto-terminates after run)
# MAGIC - **Runtime:** Databricks ML Runtime (includes MLflow, scikit-learn)
# MAGIC - **Node type:** Standard_DS3_v2 or equivalent
# MAGIC
# MAGIC ### Notifications
# MAGIC - On success: email / Slack webhook
# MAGIC - On failure: email / PagerDuty alert

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Next Steps: Deployment → Testing → Monitoring
# MAGIC
# MAGIC | Phase | Action |
# MAGIC |-------|--------|
# MAGIC | **Deployment** | Serve `nigeria_fraud_rf` via Databricks Model Serving endpoint |
# MAGIC | **Testing** | Run inference against held-out test set; validate predictions via REST API |
# MAGIC | **Monitoring** | Track prediction drift, data drift, and F1 degradation over time |
# MAGIC | **Retraining trigger** | Auto-trigger this pipeline when drift threshold is exceeded |

# COMMAND ----------
print(f"\n Pipeline complete at: {datetime.datetime.now()}")
print("Next: Deploy model via Databricks Model Serving.")

