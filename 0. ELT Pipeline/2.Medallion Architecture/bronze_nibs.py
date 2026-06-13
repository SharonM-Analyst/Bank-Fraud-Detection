from pyspark import pipelines as dp


@dp.table(
    name="bronze_nibs"
)
def bronze_nibs():

    bronze_df = spark.read.table(
        "mlops.cicdpipe.nibs"
    )

    return bronze_df