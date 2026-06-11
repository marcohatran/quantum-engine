package quantum.demo

import java.nio.file.Files

import org.apache.spark.sql.SparkSession

import quantum.spark.{QuantumExec, QuantumExtension, QuantumStrategy}

/**
 * Head-to-head: the same aggregation query over the same Parquet data,
 * once in a Quantum-enabled session and once in vanilla Spark. Both run
 * `master=local[1]`; Quantum runs FIRST so JVM warm-up bias, if any,
 * favors the vanilla baseline. Run via `scripts/run-perf-compare.sh`
 * (optionally `run-perf-compare.sh <rows> <groups>`).
 */
object QuantumPerfCompare {

  private val Warmups  = 2
  private val Measured = 5

  def main(args: Array[String]): Unit = {
    val rows   = if (args.length > 0) args(0).toLong else 2000000L
    val groups = if (args.length > 1) args(1).toLong else 200L

    val dir  = Files.createTempDirectory("quantum_perf_").toFile
    val path = new java.io.File(dir, "t.parquet").getAbsolutePath

    // Write the dataset once with a throwaway session.
    locally {
      val spark = baseBuilder().getOrCreate()
      spark.range(rows).selectExpr(s"id % $groups AS k", "id AS v")
        .coalesce(1).write.mode("overwrite").parquet(path)
      spark.stop()
    }

    val sql =
      s"""SELECT k, SUM(v) AS s, COUNT(*) AS c
         |FROM t WHERE v > ${rows / 10} GROUP BY k""".stripMargin

    println(s"rows=$rows groups=$groups  warmups=$Warmups measured=$Measured")
    println(s"query: ${sql.replace('\n', ' ')}\n")

    val (quantumTimes, quantumSum) = measure(path, sql, withQuantum = true)
    val (vanillaTimes, vanillaSum) = measure(path, sql, withQuantum = false)

    require(quantumSum == vanillaSum,
      s"result mismatch: quantum=$quantumSum vanilla=$vanillaSum")
    println(s"\nresult checksum identical: $quantumSum")

    def stats(ts: Seq[Double]): String = {
      val sorted = ts.sorted
      f"median ${sorted(sorted.size / 2)}%8.1f ms   min ${sorted.head}%8.1f ms   " +
        f"all [${ts.map(t => f"$t%.0f").mkString(", ")}]"
    }
    println(f"quantum: ${stats(quantumTimes)}")
    println(f"vanilla: ${stats(vanillaTimes)}")
    val speedup = vanillaTimes.sorted.apply(vanillaTimes.size / 2) /
                  quantumTimes.sorted.apply(quantumTimes.size / 2)
    println(f"\nmedian speedup (vanilla / quantum): $speedup%.2fx")

    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf); f.delete()
    }
    rmRf(dir)
  }

  private def baseBuilder() = SparkSession.builder()
    .appName("quantum-perf")
    .master("local[1]")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.sql.optimizer.excludedRules",
            "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation")

  /** Returns (measured times in ms, checksum of results). */
  private def measure(path: String, sql: String,
                      withQuantum: Boolean): (Seq[Double], Long) = {
    val builder = baseBuilder()
    val spark   = (if (withQuantum) builder.withExtensions(new QuantumExtension())
                   else builder).getOrCreate()
    try {
      spark.read.parquet(path).createOrReplaceTempView("t")

      def runOnce(): (Double, Long) = {
        QuantumStrategy.reset()
        val t0   = System.nanoTime()
        val rows = spark.sql(sql).collect()
        val ms   = (System.nanoTime() - t0) / 1e6
        val sum  = rows.map(r => r.getLong(0) * 31 + r.getLong(1) + r.getLong(2)).sum
        (ms, sum)
      }

      (1 to Warmups).foreach(_ => runOnce())
      val routed =
        if (withQuantum) QuantumStrategy.translations.exists(_.startsWith("HashAggregate("))
        else !QuantumStrategy.translations.exists(_.startsWith("HashAggregate("))
      val label = if (withQuantum) "quantum" else "vanilla"
      require(routed, s"$label session routed unexpectedly: " +
        QuantumStrategy.translations)
      println(s"[$label] routing verified" +
        (if (withQuantum) s": ${QuantumStrategy.lastTranslation.getOrElse("")}" else ""))
      if (withQuantum) {
        println(s"[$label] native trace: " +
          Option(QuantumExec.lastTraceJson.get()).getOrElse("(none)"))
      }

      val results = (1 to Measured).map(_ => runOnce())
      (results.map(_._1), results.head._2)
    } finally {
      spark.stop()
    }
  }
}
