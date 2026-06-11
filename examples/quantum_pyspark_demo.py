"""Quantum Engine from PySpark.

The engine is JVM/native-side, so PySpark needs no special bindings:
the extension jar goes on the driver classpath, the native library path
into the driver JVM options, and every SQL query that the strategy can
translate runs on the C++ engine. Run via scripts/run-pyspark-demo.sh.
"""
import os

from pyspark.sql import SparkSession

JARS = os.environ["QUANTUM_JARS"]          # quantum jar + runtime deps
NATIVE_DIR = os.environ["QUANTUM_NATIVE_DIR"]

spark = (
    SparkSession.builder
    .appName("quantum-pyspark-demo")
    .master("local[1]")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.sql.extensions", "quantum.spark.QuantumExtension")
    .config("spark.driver.extraClassPath", JARS)
    .config("spark.driver.extraJavaOptions",
            f"-Djna.library.path={NATIVE_DIR} "
            "-Dio.netty.tryReflectionSetAccessible=true")
    # Keep Filters visible to the planner strategy (same as the JVM tests).
    .config("spark.sql.optimizer.excludedRules",
            "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# PySpark's createDataFrame builds an RDD-backed plan the strategy can't
# peel — go through Parquet instead, which routes via the native
# FileScanNode (and is the representative production path anyway).
import tempfile

data_dir = tempfile.mkdtemp(prefix="quantum_pyspark_")
spark.createDataFrame(
    [
        ("EMEA", "laptop",   1200.0),
        ("EMEA", "phone",     800.0),
        ("AMER", "laptop",   1500.0),
        ("AMER", "tablet",     90.0),   # filtered out below
        ("AMER", "phone",     950.0),
        ("APAC", "laptop",   1100.0),
        ("APAC", "monitor",   400.0),
        ("EMEA", "keyboard",   60.0),   # filtered out below
    ],
    ["region", "product", "amount"],
).coalesce(1).write.mode("overwrite").parquet(data_dir + "/sales")
spark.read.parquet(data_dir + "/sales").createOrReplaceTempView("sales")

query = spark.sql("""
    SELECT region, COUNT(*) AS orders, SUM(amount) AS revenue
    FROM sales
    WHERE amount > 100
    GROUP BY region
    ORDER BY revenue DESC
""")
rows = query.collect()

print("=== Results ===")
for r in rows:
    print(f"  {r.region:<6} orders={r.orders} revenue={r.revenue:.2f}")

plan = query._jdf.queryExecution().executedPlan().toString()
print("\n=== Physical plan (note the Quantum node) ===")
print("\n".join(plan.splitlines()[:3]))

jvm = spark.sparkContext._jvm
print("\n=== Native translation ===")
print(" ", jvm.quantum.spark.QuantumStrategy.lastTranslation().getOrElse(None))

print("\n=== Native per-operator trace ===")
print(" ", jvm.quantum.spark.QuantumExec.lastTraceJson().get())

print("\n=== Engine counters (Prometheus) ===")
print(jvm.quantum.QuantumEngine.metricsPrometheus(), end="")

assert "Quantum" in plan, "expected the native Quantum node in the plan!"
spark.stop()
import shutil
shutil.rmtree(data_dir, ignore_errors=True)
print("\nOK: query executed on the native engine.")
