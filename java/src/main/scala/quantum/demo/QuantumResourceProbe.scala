package quantum.demo

import java.lang.management.ManagementFactory

import scala.jdk.CollectionConverters._

import org.apache.spark.sql.SparkSession

import quantum.spark.QuantumExtension

/**
 * Single-mode probe for resource comparison; one JVM per mode so the
 * wrapping script can measure the whole process with /usr/bin/time -l.
 *
 *   args: <prep|quantum|vanilla> <dataDir> [rows] [groups] [iterations]
 *
 * prep writes the parquet dataset; quantum/vanilla run the aggregation
 * query (1 warmup + N measured) and print wall times, JVM heap in use,
 * and GC totals. Run via scripts/run-resource-compare.sh.
 */
object QuantumResourceProbe {
  def main(args: Array[String]): Unit = {
    val mode    = args(0)
    val dataDir = args(1)
    val rows    = if (args.length > 2) args(2).toLong else 10000000L
    val groups  = if (args.length > 3) args(3).toLong else 1000L
    val iters   = if (args.length > 4) args(4).toInt else 3
    val query   = if (args.length > 5) args(5) else "agg"

    val builder = SparkSession.builder()
      .appName(s"quantum-resource-$mode")
      .master("local[1]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "1")
      .config("spark.sql.optimizer.excludedRules",
              "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation")
      // Measure the engine on wide results too (the production default
      // gates these to Spark; see spark.quantum.plain.maxBytes).
      .config("spark.quantum.plain.maxBytes", Long.MaxValue.toString)
    val spark =
      (if (mode == "quantum") builder.withExtensions(new QuantumExtension())
       else builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    if (mode == "prep") {
      // Optional 5th arg: extra double columns, for wide-table scans where
      // the queries below still touch only k and v (column pruning).
      // Optional 6th arg: parquet block (row group) size in bytes, for
      // multi-row-group files (row-group statistics skipping).
      val extraCols = if (args.length > 4) args(4).toInt else 0
      val extras = (0 until extraCols).map(i => s"CAST(id * 1.0 + $i AS DOUBLE) AS e$i")
      val writer = spark.range(rows)
        .selectExpr(Seq(s"id % $groups AS k", "id AS v") ++ extras: _*)
        .coalesce(1).write.mode("overwrite")
      (if (args.length > 5) writer.option("parquet.block.size", args(5).toLong)
       else writer).parquet(dataDir)
      spark.stop()
      return
    }

    spark.read.parquet(dataDir).createOrReplaceTempView("t")
    // "agg": small result (one row per group). "select": wide result --
    // ~90% of input rows return to the driver, stressing result conversion.
    val (sql, expectRows) =
      if (query == "select")
        (s"SELECT k, v FROM t WHERE v > ${rows / 10}", rows - rows / 10 - 1)
      else
        (s"SELECT k, SUM(v) AS s, COUNT(*) AS c FROM t " +
         s"WHERE v > ${rows / 10} GROUP BY k", groups)

    def runOnce(): Double = {
      val t0 = System.nanoTime()
      val n  = spark.sql(sql).collect().length
      require(n == expectRows, s"expected $expectRows rows, got $n")
      (System.nanoTime() - t0) / 1e6
    }

    runOnce() // warmup
    val times = (1 to iters).map(_ => runOnce())

    val heapUsedMb =
      (Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()) / 1e6
    val gc       = ManagementFactory.getGarbageCollectorMXBeans.asScala
    val gcCount  = gc.map(_.getCollectionCount).sum
    val gcTimeMs = gc.map(_.getCollectionTime).sum

    // Post-run floor: RSS after a GC, with all sessions' queries done —
    // shows what the pooled engine + allocators retain between queries.
    System.gc(); Thread.sleep(200)
    val finalRssMb = {
      val pid = ProcessHandle.current().pid()
      val p   = new ProcessBuilder("ps", "-o", "rss=", "-p", pid.toString).start()
      val out = new String(p.getInputStream.readAllBytes()).trim
      p.waitFor()
      if (out.nonEmpty) out.toLong / 1024 else -1L
    }

    println(f"RESULT mode=$mode wall_ms=${times.sorted.apply(times.size / 2)}%.1f " +
            f"heap_used_mb=$heapUsedMb%.0f gc_count=$gcCount gc_time_ms=$gcTimeMs " +
            f"final_rss_mb=$finalRssMb")
    spark.stop()
  }
}
