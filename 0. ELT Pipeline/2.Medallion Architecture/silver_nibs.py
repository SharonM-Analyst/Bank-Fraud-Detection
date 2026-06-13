from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import LongType



@dp.table(
    name="silver_nibs"
)
def silver_nibs():

    df = spark.read.table("LIVE.bronze_nibs")
    
    # Initialize silver_df from df
    silver_df = df

    # Columns to exclude from bigint conversion
    exclude_columns = [
        'transaction_id',
        'customer_id',
        'channel'
        'merchant_category',
        'bank',
        'location',
        'age_group',
        'is_weekend',
        'is_peak_hour',
        'timestamp'
    ]


    # Replace NULL fraud techniques
    silver_df = silver_df.withColumn(
        'fraud_technique',
        F.when(
            F.col('fraud_technique').isNull(),
            'unknown'
        ).otherwise(F.col('fraud_technique'))
    )

    # Convert numeric columns
    all_columns = df.columns

    for col_name in all_columns:

        if (
            col_name not in exclude_columns
            and col_name != 'fraud_technique'
            and col_name != '_rescued_data'
            
        ):

            silver_df = silver_df.withColumn(
                col_name,
                F.round(
                    F.col(col_name) * 100,
                    0
                ).cast(LongType())
            )


    

    # Drop unnecessary columns
    silver_df = silver_df.drop(
        '_rescued_data'
    )

    # Remove duplicates
    silver_df = silver_df.dropDuplicates()

    return silver_df