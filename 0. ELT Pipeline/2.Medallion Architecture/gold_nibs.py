from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


# =========================================================
# GOLD LAYER
# =========================================================

@dp.table(
    name="gold_nibs"
)
def gold_nibs():

    gold_df = spark.read.table("LIVE.silver_nibs")

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    numeric_columns = [
        'amount','hour','day_of_week','month','tx_count_24h',
        'amount_sum_24h','amount_mean_7d','amount_std_7d',
        'tx_count_total','amount_mean_total','amount_std_total',
        'channel_diversity','location_diversity',
        'amount_vs_mean_ratio','online_channel_ratio','is_fraud',
        'hour_sin','hour_cos','day_sin','day_cos','month_sin',
        'month_cos','amount_log','amount_rounded',
        'velocity_score','merchant_risk_score','composite_risk'
    ]

    # Convert bigint back to decimal
    for col_name in numeric_columns:
        if col_name in gold_df.columns:
            gold_df = gold_df.withColumn(
                col_name,
                (F.col(col_name) / 100.0).cast(DoubleType())
            )

    # Handle NULL numeric values
    for col_name in numeric_columns:
        if col_name in gold_df.columns:

            quantiles = gold_df.stat.approxQuantile(col_name, [0.5], 0.01)
            median_val = quantiles[0] if len(quantiles) > 0 else 0

            gold_df = gold_df.withColumn(
                col_name,
                F.when(F.col(col_name).isNull(), median_val)
                 .otherwise(F.col(col_name))
            )



    # =====================================================
    # CATEGORICAL NULL HANDLING
    # =====================================================

    categorical_columns = [
        'merchant_category','bank','location','age_group'
    ]

    for col_name in categorical_columns:
        if col_name in gold_df.columns:

            mode_row = (
                gold_df.groupBy(col_name)
                .count()
                .orderBy(F.desc("count"))
                .first()
            )

            mode_value = mode_row[0] if mode_row else "Unknown"

            gold_df = gold_df.withColumn(
                col_name,
                F.when(F.col(col_name).isNull(), mode_value)
                 .otherwise(F.col(col_name))
            )

    # =====================================================
    # RISK CATEGORY
    # =====================================================

    gold_df = gold_df.withColumn(
        "risk_category",
        F.when(F.col("composite_risk") >= 0.75, "High Risk")
         .when(F.col("composite_risk") >= 0.50, "Medium Risk")
         .when(F.col("composite_risk") >= 0.25, "Low Risk")
         .otherwise("Very Low Risk")
    )

    # =====================================================
    # FRAUD FLAG
    # =====================================================

    gold_df = gold_df.withColumn(
        "is_fraudulent",
        F.when(F.col("is_fraud") == 1, True).otherwise(False)
    )

    # =====================================================
    # AUDIT COLUMNS
    # =====================================================

    gold_df = gold_df.withColumn("gold_processed_timestamp", F.current_timestamp())
    gold_df = gold_df.withColumn("data_quality_score", F.lit(100.0))

    # =====================================================
    # FEATURES
    # =====================================================

    gold_df = gold_df.withColumn(
        "transaction_hour_category",
        F.when(F.col("hour").between(6,11), "Morning")
         .when(F.col("hour").between(12,17), "Afternoon")
         .when(F.col("hour").between(18,22), "Evening")
         .otherwise("Night")
    )

    gold_df = gold_df.withColumn(
        "amount_category",
        F.when(F.col("amount") < 100, "Small")
         .when(F.col("amount") < 500, "Medium")
         .when(F.col("amount") < 2000, "Large")
         .otherwise("Very Large")
    )

    # =====================================================
    # COLUMN ORDER
    # =====================================================

    column_order = [
        'transaction_id',
        'customer_id',
        'timestamp',
        'amount',
        'amount_category',
        'is_fraud',
        'is_fraudulent',
        'fraud_technique',
        'risk_category',
        'composite_risk',
        'channel',
        'merchant_category',
        'bank',
        'location',
        'age_group',
        'transaction_hour_category',
        'month',
        'hour',
        'day_of_week',
        'is_weekend',
        'is_peak_hour'
    ]

    remaining_cols = [
        c for c in gold_df.columns if c not in column_order
    ]

    gold_df = gold_df.select(*(column_order + remaining_cols))

    return gold_df