package quantum.demo

import org.apache.spark.sql.SparkSession

import quantum.QuantumEngine
import quantum.spark.{QuantumExec, QuantumExtension, QuantumStrategy}

/**
 * Minimal end-to-end job: a Spark session with the Quantum extension runs
 * a SQL aggregation that executes on the native engine, then prints the
 * results plus the engine-side diagnostics (translation label, per-operator
 * trace, Prometheus counters).
 *
 * Run it with `scripts/run-demo.sh` from the repo root (builds nothing
 * itself — the native lib must exist: `cmake --build --preset=debug`).
 */
object QuantumDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("quantum-demo")
      .master("local[1]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "1")
      // Keep the Filter visible to the planner strategy (see tests).
      .config("spark.sql.optimizer.excludedRules",
              "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation")
      .withExtensions(new QuantumExtension())
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    // A small sales table.
    Seq(
      ("EMEA", "laptop",   1200.0),
      ("EMEA", "phone",     800.0),
      ("AMER", "laptop",   1500.0),
      ("AMER", "tablet",     90.0),  // filtered out below
      ("AMER", "phone",     950.0),
      ("APAC", "laptop",   1100.0),
      ("APAC", "monitor",   400.0),
      ("EMEA", "keyboard",   60.0)   // filtered out below
    ).toDF("region", "product", "amount").createOrReplaceTempView("sales")

    val query = spark.sql("""
      SELECT region, COUNT(*) AS orders, SUM(amount) AS revenue
      FROM sales
      WHERE amount > 100
      GROUP BY region
      ORDER BY revenue DESC
    """)
    val rows = query.collect()

    println("=== Results ===")
    rows.foreach(r => println(f"  ${r.getString(0)}%-6s orders=${r.getLong(1)} revenue=${r.getDouble(2)}%.2f"))

    println("\n=== Physical plan (note the Quantum node) ===")
    println(query.queryExecution.executedPlan.toString.linesIterator
      .take(3).mkString("\n"))

    println("\n=== Native translation ===")
    println("  " + QuantumStrategy.lastTranslation.getOrElse("(none — fell back to Spark)"))

    println("\n=== Native per-operator trace ===")
    println("  " + Option(QuantumExec.lastTraceJson.get()).getOrElse("(none)"))

    println("\n=== Engine counters (Prometheus) ===")
    print(QuantumEngine.metricsPrometheus())

    spark.stop()
  }
}
