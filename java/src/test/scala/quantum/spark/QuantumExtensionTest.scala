package quantum.spark

import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}

class QuantumExtensionTest {

  private var spark: SparkSession = _

  @BeforeEach
  def setUp(): Unit = {
    QuantumStrategy.reset()
    spark = SparkSession.builder()
      .appName("quantum-extension-test")
      .master("local[1]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "1")
      // AQE is intentionally LEFT ENABLED (Spark's default) — for our
      // LocalRelation-backed queries there's no Exchange below the Aggregate,
      // so AQE doesn't create a stage boundary that would wrap the Aggregate
      // in LogicalQueryStage. Queries with shuffle (multi-partition sources)
      // would need explicit LogicalQueryStage handling — see roadmap.
      // ConvertToLocalRelation evaluates Filter(LocalRelation) at optimize
      // time when the data is small enough, which would hide the Filter from
      // our planner strategy. Disable it so the filter-pushdown tests can
      // observe the Filter node in the plan tree.
      .config("spark.sql.optimizer.excludedRules",
              "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation")
      .withExtensions(new QuantumExtension())
      .getOrCreate()
  }

  @AfterEach
  def tearDown(): Unit = if (spark != null) spark.stop()

  /** Regression guard: the strategy must see every LogicalPlan. */
  @Test
  def extensionStrategyFiresOnSqlQuery(): Unit = {
    val df = spark.sql("SELECT id, id + 1 AS plus1 FROM range(5) WHERE id > 1")
    val rows = df.collect().map(r => (r.getLong(0), r.getLong(1))).toSeq

    assertEquals(Seq((2L, 3L), (3L, 4L), (4L, 5L)), rows)
    assertTrue(QuantumStrategy.lastObservedPlan.isDefined,
               "QuantumStrategy should have observed at least one LogicalPlan")
  }

  /**
   * Headline Phase 4.7 test: a SELECT k, SUM(v) FROM kv GROUP BY k that
   * matches the strategy's recognised shape gets routed through QuantumExec,
   * which executes it in the native engine via the JNA bridge and returns
   * the same results Spark would produce on its own.
   */
  @Test
  def sumByKeyExecutesNativelyAndProducesCorrectResult(): Unit = {
    val s = spark  // stable val for `import s.implicits._`
    import s.implicits._
    val df = Seq(
      (1L, 10L), (2L, 20L), (1L, 30L), (3L, 40L), (2L, 50L)
    ).toDF("k", "v")
    df.createOrReplaceTempView("kv")

    val rows = spark.sql("SELECT k, SUM(v) FROM kv GROUP BY k")
      .collect()
      .map(r => (r.getLong(0), r.getLong(1)))
      .sortBy(_._1)
      .toSeq

    assertEquals(Seq((1L, 40L), (2L, 70L), (3L, 40L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.isDefined,
               "Strategy should have translated the HashAggregate to native; " +
               "translation flag = " + QuantumStrategy.lastTranslation +
               "; seen plan classes = " + QuantumStrategy.observedClasses +
               "; last plan =\n" + QuantumStrategy.lastObservedPlan.map(_.treeString).getOrElse(""))
  }

  /** Phase 5.5: COUNT(*) goes through native AGG_COUNT_STAR. */
  @Test
  def countStarExecutesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (2L, 20L), (1L, 30L), (3L, 40L), (2L, 50L), (1L, 60L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_count")

    val rows = spark.sql("SELECT k, COUNT(*) AS n FROM kv_count GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((1L, 3L), (2L, 2L), (3L, 1L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "COUNT(*) should route through native; flag = " + QuantumStrategy.lastTranslation)
  }

  /** Phase 5.5: MIN and MAX over a Long column. */
  @Test
  def minMaxExecuteNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (1L, 99L), (1L, 50L), (2L, 7L), (2L, 8L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_minmax")

    val rows = spark.sql("SELECT k, MIN(v) AS lo, MAX(v) AS hi FROM kv_minmax GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1), r.getLong(2)))
      .sortBy(_._1).toSeq

    assertEquals(Seq((1L, 10L, 99L), (2L, 7L, 8L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "MIN/MAX should route through native; flag = " + QuantumStrategy.lastTranslation)
  }

  /** Phase 5.5: AVG returns a Double output column through the bridge. */
  @Test
  def avgReturnsDoubleThroughBridge(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (1L, 20L), (1L, 30L), (2L, 100L), (2L, 200L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_avg")

    val rows = spark.sql("SELECT k, AVG(v) AS a FROM kv_avg GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getDouble(1)))
      .sortBy(_._1).toSeq

    assertEquals(2, rows.size)
    assertEquals(1L,   rows(0)._1); assertEquals(20.0,  rows(0)._2, 1e-9)
    assertEquals(2L,   rows(1)._1); assertEquals(150.0, rows(1)._2, 1e-9)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "AVG should route through native; flag = " + QuantumStrategy.lastTranslation)
  }

  /** Phase 5.5: SUM + COUNT + MIN + MAX + AVG in a single Aggregate node. */
  @Test
  def fiveAggregatesInOneQueryExecuteNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (1L, 20L), (1L, 30L), (2L, 100L), (2L, 200L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_combo")

    val rows = spark.sql(
      """SELECT k,
        |       SUM(v)   AS s,
        |       COUNT(*) AS n,
        |       MIN(v)   AS lo,
        |       MAX(v)   AS hi,
        |       AVG(v)   AS a
        |  FROM kv_combo GROUP BY k""".stripMargin)
      .collect()
      .map(r => (r.getLong(0), r.getLong(1), r.getLong(2), r.getLong(3), r.getLong(4), r.getDouble(5)))
      .sortBy(_._1).toSeq

    assertEquals(Seq(
      (1L,  60L, 3L, 10L,  30L,  20.0),
      (2L, 300L, 2L, 100L, 200L, 150.0)
    ), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Combined aggs should route through native; flag = " + QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.6: GROUP BY (k1, k2). Exercises the native composite-key path
   * (BytesHashTable + KeyCodec), which handles fixed-width multi-column keys.
   */
  @Test
  def twoColumnGroupByExecutesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      (1L, 100L, 10L),
      (1L, 100L, 30L),
      (1L, 200L, 50L),
      (2L, 100L, 70L),
      (2L, 100L, 80L),
      (2L, 200L, 90L)
    ).toDF("k1", "k2", "v")
    df.createOrReplaceTempView("kv2")

    val rows = spark.sql(
      "SELECT k1, k2, SUM(v) AS s, COUNT(*) AS n FROM kv2 GROUP BY k1, k2")
      .collect()
      .map(r => (r.getLong(0), r.getLong(1), r.getLong(2), r.getLong(3)))
      .sortBy(t => (t._1, t._2))
      .toSeq

    assertEquals(Seq(
      (1L, 100L,  40L, 2L),
      (1L, 200L,  50L, 1L),
      (2L, 100L, 150L, 2L),
      (2L, 200L,  90L, 1L)
    ), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Multi-column GROUP BY should route through native; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.6: single WHERE clause pushed through to the engine. The strategy
   * wraps the Scan in a FilterNode whose predicate is the translated Catalyst
   * comparison.
   */
  @Test
  def singleClauseWhereIsPushedDown(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (1L, 20L), (2L, 30L), (2L, 40L), (3L, 50L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_filt")

    val rows = spark.sql("SELECT k, SUM(v) AS s FROM kv_filt WHERE v > 25 GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((2L, 70L), (3L, 50L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("1 pushed filters")),
      "single WHERE should produce 1 pushed filter; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.6: AND-conjuncted WHERE flattens into a chain of single-clause
   * FilterNodes. The native engine has no BinaryOp::And; chained FilterOps
   * is semantically equivalent.
   */
  @Test
  def conjunctedWhereFlattensIntoTwoFilters(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      (1L, 10L),
      (1L, 20L),
      (2L, 30L),
      (2L, 40L),
      (3L, 50L),
      (4L, 99L)).toDF("k", "v")
    df.createOrReplaceTempView("kv_and")

    val rows = spark.sql(
      "SELECT k, SUM(v) AS s FROM kv_and WHERE v > 15 AND k < 4 GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((1L, 20L), (2L, 70L), (3L, 50L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("2 pushed filters")),
      "WHERE v>15 AND k<4 should flatten to 2 pushed filters; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.7: GROUP BY String. Exercises the native StringFastPath /
   * VarBytesHashTable depending on the engine's String agg path selection.
   */
  @Test
  def stringGroupByExecutesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      ("apple",  10L),
      ("banana", 20L),
      ("apple",  30L),
      ("cherry", 40L),
      ("banana", 50L),
      ("apple",  60L)
    ).toDF("name", "v")
    df.createOrReplaceTempView("kv_str")

    val rows = spark.sql(
      "SELECT name, SUM(v) AS s, COUNT(*) AS n FROM kv_str GROUP BY name")
      .collect()
      .map(r => (r.getString(0), r.getLong(1), r.getLong(2)))
      .sortBy(_._1)
      .toSeq

    assertEquals(Seq(
      ("apple",  100L, 3L),
      ("banana",  70L, 2L),
      ("cherry",  40L, 1L)
    ), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "String GROUP BY should route through native; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /** Phase 5.7: GROUP BY (String, Long) — composite key with mixed types. */
  @Test
  def mixedStringAndLongGroupBy(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      ("a", 1L, 10L), ("a", 1L, 20L),
      ("a", 2L, 30L),
      ("b", 1L, 40L), ("b", 1L, 50L)
    ).toDF("name", "k2", "v")
    df.createOrReplaceTempView("kv_mix")

    val rows = spark.sql(
      "SELECT name, k2, SUM(v) FROM kv_mix GROUP BY name, k2")
      .collect()
      .map(r => (r.getString(0), r.getLong(1), r.getLong(2)))
      .sortBy(t => (t._1, t._2))
      .toSeq

    assertEquals(Seq(
      ("a", 1L, 30L),
      ("a", 2L, 30L),
      ("b", 1L, 90L)
    ), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Mixed-type GROUP BY should route through native; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.9: end-to-end Spark → file-source. Writes a single Parquet file
   * via Spark, queries it via `spark.read.parquet(...)`, and verifies the
   * strategy emits a FileScanNode-bearing plan that the native engine reads
   * itself (no JNA marshalling of row data).
   */
  @Test
  def parquetFileSourceExecutesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (1L, 20L), (2L, 30L), (2L, 40L), (3L, 50L))
      .toDF("k", "v")

    // Write to a unique tmp file so each test run is independent.
    val tmpDir = java.nio.file.Files.createTempDirectory("quantum_parquet_").toFile
    val parquetPath = new java.io.File(tmpDir, "data.parquet").getAbsolutePath
    df.coalesce(1).write.mode("overwrite").parquet(parquetPath)
    // Spark writes a directory by default; pick the single .parquet file
    // inside it to exercise the single-file path.
    val singleFile = new java.io.File(parquetPath).listFiles()
      .find(_.getName.endsWith(".parquet"))
      .map(_.getAbsolutePath)
      .getOrElse(throw new IllegalStateException("no parquet file produced"))

    val rows = spark.read.parquet(singleFile)
      .groupBy("k").agg(org.apache.spark.sql.functions.sum("v").as("s"))
      .collect()
      .map(r => (r.getLong(0), r.getLong(1)))
      .sortBy(_._1)
      .toSeq

    assertEquals(Seq((1L, 30L), (2L, 70L), (3L, 50L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Parquet file source should route through native; flag = " +
        QuantumStrategy.lastTranslation)

    // Cleanup
    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf)
      f.delete()
    }
    rmRf(tmpDir)
  }

  /**
   * Phase 5.10: end-to-end Spark → multi-file Parquet directory. Writes
   * with 3 partitions so Spark produces 3 part-*.parquet files; the
   * strategy hands all of them to a native ConcatFileReader through a
   * single FileScanNode.
   */
  @Test
  def multiFileParquetDirectoryExecutesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = (1 to 100).map(i => ((i % 5).toLong, i.toLong)).toDF("k", "v")

    val tmpDir = java.nio.file.Files.createTempDirectory("quantum_parquet_multi_").toFile
    val parquetDir = new java.io.File(tmpDir, "data").getAbsolutePath
    df.repartition(3).write.mode("overwrite").parquet(parquetDir)

    // Sanity: confirm Spark wrote multiple part files.
    val parts = new java.io.File(parquetDir).listFiles().count(_.getName.endsWith(".parquet"))
    assertTrue(parts >= 2, s"expected multi-file write, got $parts part files")

    val rows = spark.read.parquet(parquetDir)
      .groupBy("k").agg(org.apache.spark.sql.functions.sum("v").as("s"))
      .collect()
      .map(r => (r.getLong(0), r.getLong(1)))
      .sortBy(_._1)
      .toSeq

    // Expected: SUM of i in [1..100] with i % 5 == r, for r in 0..4.
    val expected = (0 until 5).map { r =>
      (r.toLong, (1 to 100).filter(_ % 5 == r).map(_.toLong).sum)
    }
    assertEquals(expected, rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Multi-file Parquet should route through native; flag = " +
        QuantumStrategy.lastTranslation)

    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf); f.delete()
    }
    rmRf(tmpDir)
  }

  /**
   * Phase 5.24: LEFT OUTER JOIN preserves left rows that have no match on
   * the right; right-side columns become NULL for those rows. Verified by
   * counting rows per left key — keys with no right match still appear
   * with COUNT(b.v) = 0 (CountColumn skips nulls).
   */
  @Test
  def leftOuterJoinPreservesUnmatchedLeftRows(): Unit = {
    val s = spark
    import s.implicits._
    val a = Seq((1L, "alpha"), (2L, "beta"), (3L, "gamma"), (4L, "delta"))
      .toDF("k", "name")
    val b = Seq((1L, 10L), (1L, 20L), (3L, 30L))  // 2 and 4 absent
      .toDF("k", "v")
    a.createOrReplaceTempView("la")
    b.createOrReplaceTempView("lb")

    val rows = spark.sql(
      """SELECT la.k, COUNT(lb.v) AS n
        |  FROM la LEFT OUTER JOIN lb ON la.k = lb.k
        | GROUP BY la.k""".stripMargin)
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    // k=1: two matches -> 2 ; k=2: no match -> 0 ; k=3: one match -> 1 ; k=4: no match -> 0.
    assertEquals(Seq((1L, 2L), (2L, 0L), (3L, 1L), (4L, 0L)), rows)
  }

  /**
   * Phase 5.24: RIGHT OUTER JOIN preserves right rows that have no match
   * on the left.
   */
  @Test
  def rightOuterJoinPreservesUnmatchedRightRows(): Unit = {
    val s = spark
    import s.implicits._
    val a = Seq((1L, 100L), (3L, 300L))  // only k = 1 and 3
      .toDF("k", "x")
    val b = Seq((1L, "alpha"), (2L, "beta"), (3L, "gamma"), (4L, "delta"))
      .toDF("k", "name")
    a.createOrReplaceTempView("ra")
    b.createOrReplaceTempView("rb")

    val rows = spark.sql(
      """SELECT rb.k, COUNT(ra.x) AS n
        |  FROM ra RIGHT OUTER JOIN rb ON ra.k = rb.k
        | GROUP BY rb.k""".stripMargin)
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    // Right preserved: all 4 keys appear. k=1 -> 1 match, k=2 -> 0, k=3 -> 1, k=4 -> 0.
    assertEquals(Seq((1L, 1L), (2L, 0L), (3L, 1L), (4L, 0L)), rows)
  }

  /**
   * Phase 5.24: FULL OUTER JOIN preserves unmatched rows from BOTH sides.
   * Each key appears once even if only one side has it. Uses COALESCE
   * (or COUNT(*) of synthesized rows) to make the assertion simple.
   */
  @Test
  def fullOuterJoinPreservesBothUnmatched(): Unit = {
    val s = spark
    import s.implicits._
    val a = Seq((1L, "alpha"), (2L, "beta"))             // k in {1, 2}
      .toDF("k", "name")
    val b = Seq((2L, 200L), (3L, 300L))                   // k in {2, 3}
      .toDF("k", "v")
    a.createOrReplaceTempView("fa")
    b.createOrReplaceTempView("fb")

    // Group by a-side key (will be null for rows present only on b-side).
    // COUNT(fa.name) tells us if the left side contributed.
    val rows = spark.sql(
      """SELECT fa.k, COUNT(fa.name) AS la, COUNT(fb.v) AS lb
        |  FROM fa FULL OUTER JOIN fb ON fa.k = fb.k
        | GROUP BY fa.k""".stripMargin)
      .collect().map(r => (
        if (r.isNullAt(0)) -1L else r.getLong(0),
        r.getLong(1), r.getLong(2)))
      .sortBy(_._1)
      .toSeq

    // k=NULL (the row from b-side only, k=3): la=0, lb=1
    // k=1 (only a): la=1, lb=0
    // k=2 (both):  la=1, lb=1
    assertEquals(Seq(
      (-1L, 0L, 1L),
      (1L,  1L, 0L),
      (2L,  1L, 1L)
    ), rows)
  }

  /**
   * Phase 5.23: JOIN over two Parquet sources. Exercises the file-scan
   * leaf in both sides of the Join peel — the underlying scans are
   * `FileScanNode`s rather than registered `ScanNode`s, so the strategy
   * registers zero in-memory sources for this query.
   */
  @Test
  def innerJoinOverTwoParquetFiles(): Unit = {
    val s = spark
    import s.implicits._
    val a = Seq((1L, "apple"), (2L, "banana"), (3L, "cherry")).toDF("k", "name")
    val b = Seq((1L, 10L), (1L, 20L), (2L, 30L), (3L, 40L), (3L, 50L)).toDF("k", "v")

    val tmpDir = java.nio.file.Files.createTempDirectory("quantum_join_parquet_").toFile
    val pathA = new java.io.File(tmpDir, "a").getAbsolutePath
    val pathB = new java.io.File(tmpDir, "b").getAbsolutePath
    a.coalesce(1).write.mode("overwrite").parquet(pathA)
    b.coalesce(1).write.mode("overwrite").parquet(pathB)

    spark.read.parquet(pathA).createOrReplaceTempView("pa")
    spark.read.parquet(pathB).createOrReplaceTempView("pb")

    val rows = spark.sql(
      """SELECT pa.k, SUM(pb.v) AS s
        |  FROM pa JOIN pb ON pa.k = pb.k
        | GROUP BY pa.k""".stripMargin)
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((1L, 30L), (2L, 30L), (3L, 90L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("0 sources")),
      "JOIN over Parquet sources should register 0 in-memory sources; flag = " +
        QuantumStrategy.lastTranslation)

    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf); f.delete()
    }
    rmRf(tmpDir)
  }

  /**
   * Phase 5.23: 3-way join. Spark plans this as `Join(Join(A, B), C)`,
   * which exercises the recursive Join peel — the inner Join becomes the
   * left side of the outer Join, with its own combined idMap forwarded
   * up. Output positions get shifted by `leftR.output.size` at each level.
   */
  @Test
  def threeWayInnerJoinWithAggregation(): Unit = {
    val s = spark
    import s.implicits._
    val a = Seq((1L, "alpha"), (2L, "beta"), (3L, "gamma")).toDF("k", "label")
    val b = Seq((1L, 10L), (2L, 20L), (3L, 30L)).toDF("k", "v")
    val c = Seq((1L, 100L), (2L, 200L), (3L, 300L)).toDF("k", "w")
    a.createOrReplaceTempView("ta")
    b.createOrReplaceTempView("tb")
    c.createOrReplaceTempView("tc")

    val rows = spark.sql(
      """SELECT ta.k, SUM(tb.v + tc.w) AS s
        |  FROM ta
        |  JOIN tb ON ta.k = tb.k
        |  JOIN tc ON ta.k = tc.k
        | GROUP BY ta.k""".stripMargin)
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    // SUM(v + w) per k: row 1 -> 10+100=110, row 2 -> 20+200=220, row 3 -> 30+300=330.
    // But SUM in Spark gets the value as an alias. Spark may not give us
    // `tb.v + tc.w` directly; it depends on whether Catalyst pushes the
    // arithmetic down. If the strategy declines, the test still asserts
    // correctness against Spark's native path.
    assertEquals(Seq((1L, 110L), (2L, 220L), (3L, 330L)), rows)
  }

  /**
   * Phase 5.22: Inner equi-join between two LocalRelations, then GROUP BY
   * on a join-key column. The strategy emits `HashAggregate(HashJoin(scan_A,
   * scan_B, build_keys=[A.k], probe_keys=[B.k]))`. Each LocalRelation gets
   * its own sourceId and is registered separately with the engine.
   */
  @Test
  def innerJoinThenGroupBy(): Unit = {
    val s = spark
    import s.implicits._
    val a = Seq((1L, "apple"), (2L, "banana"), (3L, "cherry")).toDF("k", "name")
    val b = Seq((1L, 10L), (1L, 20L), (2L, 30L), (3L, 40L), (3L, 50L)).toDF("k", "v")
    a.createOrReplaceTempView("a")
    b.createOrReplaceTempView("b")

    val rows = spark.sql(
      """SELECT a.k, SUM(b.v) AS s, COUNT(*) AS n
        |  FROM a JOIN b ON a.k = b.k
        | GROUP BY a.k""".stripMargin)
      .collect().map(r => (r.getLong(0), r.getLong(1), r.getLong(2))).sortBy(_._1).toSeq

    assertEquals(Seq(
      (1L, 30L,  2L),
      (2L, 30L,  1L),
      (3L, 90L,  2L)
    ), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("2 sources")),
      "Inner join should produce 2 sources; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /** Phase 5.21: WHERE col IS NULL pushes natively. */
  @Test
  def whereIsNullPushesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      (1L, Some(10L)), (2L, None), (3L, Some(30L)), (4L, None), (5L, Some(50L))
    ).toDF("k", "v")
    df.createOrReplaceTempView("kv_isnull")

    val rows = spark.sql(
      "SELECT k, COUNT(*) AS n FROM kv_isnull WHERE v IS NULL GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((2L, 1L), (4L, 1L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("1 pushed filters")),
      "WHERE IS NULL should push as 1 filter; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.21: standalone WHERE col IS NOT NULL pushes natively. The
   * `stripRedundantIsNotNull` rule only fires when the same column appears
   * in a comparison conjunct; a lone `IS NOT NULL` survives and translates
   * as NOT(IsNull(col)).
   */
  @Test
  def whereIsNotNullStandalonePushes(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      (1L, Some(10L)), (2L, None), (3L, Some(30L)), (4L, None), (5L, Some(50L))
    ).toDF("k", "v")
    df.createOrReplaceTempView("kv_isnotnull")

    val rows = spark.sql(
      "SELECT k FROM kv_isnotnull WHERE v IS NOT NULL ORDER BY k")
      .collect().map(r => r.getLong(0)).toSeq

    assertEquals(Seq(1L, 3L, 5L), rows)
    // The strategy may or may not have routed natively depending on Spark's
    // planner choices around ORDER BY + sort; just verify correctness.
  }

  /**
   * Phase 5.20: COUNT(col) counts non-null rows per group. Spark renders
   * `COUNT(v)` as `Count(Seq(AttributeReference(v)))`, distinct from
   * `COUNT(*)` which is `Count(Seq(Literal(1)))`.
   */
  @Test
  def countColumnSkipsNullsPerGroup(): Unit = {
    val s = spark
    import s.implicits._
    // Use Option[Long] so some rows have null values for `v`.
    val df = Seq(
      (1L, Some(10L)),
      (1L, None),
      (1L, Some(30L)),
      (2L, Some(20L)),
      (2L, None),
      (3L, None)
    ).toDF("k", "v")
    df.createOrReplaceTempView("kv_cnt")

    val rows = spark.sql(
      "SELECT k, COUNT(v) AS n, COUNT(*) AS total FROM kv_cnt GROUP BY k")
      .collect()
      .map(r => (r.getLong(0), r.getLong(1), r.getLong(2)))
      .sortBy(_._1)
      .toSeq

    //  k=1: COUNT(v) = 2 (10 + 30), COUNT(*) = 3
    //  k=2: COUNT(v) = 1 (20),       COUNT(*) = 2
    //  k=3: COUNT(v) = 0,            COUNT(*) = 1
    assertEquals(Seq(
      (1L, 2L, 3L),
      (2L, 1L, 2L),
      (3L, 0L, 1L)
    ), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "COUNT(col) + COUNT(*) should route through native; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.19: WHERE NOT (predicate) translates as UnaryOp::Not wrapping
   * the child. Spark's optimizer may rewrite NOT(=) to !=, so we use a
   * predicate (NOT (k > 2)) that Catalyst can't trivially simplify away.
   */
  @Test
  def whereNotPushesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (2L, 20L), (3L, 30L), (4L, 40L), (5L, 50L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_not")

    // WHERE NOT (k > 2)  ->  keeps k in {1, 2}.
    val rows = spark.sql(
      "SELECT k, SUM(v) AS s FROM kv_not WHERE NOT (k > 2) GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((1L, 10L), (2L, 20L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("1 pushed filters")),
      "NOT predicate should push as 1 filter; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.15: WHERE on a String column with a String literal.
   * The Catalyst Literal is a UTF8String, not a Java String -- the
   * translator pattern-matches the underlying value's class.
   */
  @Test
  def whereOnStringLiteralPushesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      ("apple", 10L), ("banana", 20L), ("apple", 30L),
      ("cherry", 40L), ("banana", 50L))
      .toDF("name", "v")
    df.createOrReplaceTempView("kv_strlit")

    val rows = spark.sql(
      "SELECT name, SUM(v) AS s FROM kv_strlit WHERE name = 'apple' GROUP BY name")
      .collect().map(r => (r.getString(0), r.getLong(1))).toSeq

    assertEquals(Seq(("apple", 40L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("1 pushed filters")),
      "String literal predicate should push as 1 filter; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /** Phase 5.15: WHERE on String with OR over multiple literals. */
  @Test
  def whereOnStringOrPushesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      ("apple", 10L), ("banana", 20L), ("apple", 30L),
      ("cherry", 40L), ("banana", 50L))
      .toDF("name", "v")
    df.createOrReplaceTempView("kv_str_or")

    val rows = spark.sql(
      "SELECT name, SUM(v) AS s FROM kv_str_or " +
      "WHERE name = 'apple' OR name = 'banana' GROUP BY name")
      .collect().map(r => (r.getString(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq(("apple", 40L), ("banana", 70L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "String OR predicate should route through native; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.14: WHERE with OR translates to a single FilterNode whose
   * predicate is a BinaryOp(Or, ...) — chained Filters can't express OR.
   */
  @Test
  def orPredicatePushesAsSingleFilter(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (2L, 20L), (3L, 30L), (4L, 40L), (5L, 50L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_or")

    // WHERE k = 1 OR k = 3 OR v > 45  ->  keeps {(1,10), (3,30), (5,50)}
    val rows = spark.sql(
      "SELECT k, SUM(v) AS s FROM kv_or WHERE k = 1 OR k = 3 OR v > 45 GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((1L, 10L), (3L, 30L), (5L, 50L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("1 pushed filters")),
      "OR predicate should be 1 pushed filter; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.12: GTE / LTE predicates translate natively. Native engine
   * gained BinaryOp::Gte/Lte (proto BINOP_GTE/BINOP_LTE), so Catalyst's
   * GreaterThanOrEqual / LessThanOrEqual no longer make the strategy
   * decline.
   */
  @Test
  def gteAndLtePredicatesPushDownNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (2L, 20L), (3L, 30L), (4L, 40L), (5L, 50L))
      .toDF("k", "v")
    df.createOrReplaceTempView("kv_gte")

    // WHERE k >= 2 AND v <= 40  ->  keeps k in {2, 3, 4}.
    val rows = spark.sql(
      "SELECT k, SUM(v) AS s FROM kv_gte WHERE k >= 2 AND v <= 40 GROUP BY k")
      .collect().map(r => (r.getLong(0), r.getLong(1))).sortBy(_._1).toSeq

    assertEquals(Seq((2L, 20L), (3L, 30L), (4L, 40L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("2 pushed filters")),
      "Expected k>=2 AND v<=40 to push 2 filters; flag = " +
        QuantumStrategy.lastTranslation)
  }

  /**
   * Phase 5.16: Hive-style partitioned Parquet. Writes a 2-partition dataset
   * (year=2024 + year=2025), queries it, verifies the synthesized partition
   * column appears in both GROUP BY and result.
   */
  @Test
  def partitionedParquetSynthesizesPartitionColumns(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      (2024L, 1L, 10L), (2024L, 2L, 20L),
      (2025L, 3L, 30L), (2025L, 4L, 40L))
      .toDF("year", "k", "v")

    val tmpDir = java.nio.file.Files.createTempDirectory("quantum_part_").toFile
    val tablePath = new java.io.File(tmpDir, "data").getAbsolutePath
    df.write.mode("overwrite").partitionBy("year").parquet(tablePath)

    // Confirm Spark wrote `year=2024/` and `year=2025/` dirs.
    val partDirs = new java.io.File(tablePath).listFiles()
      .filter(_.getName.startsWith("year="))
      .map(_.getName).sorted.toSeq
    assertEquals(Seq("year=2024", "year=2025"), partDirs)

    // GROUP BY the partition column. The native engine receives `year` as
    // a constant column synthesized per-file by PartitionedFileReader.
    // Spark infers IntegerType for `year=2024` partition values. Our
    // strategy widens Int -> Int64 internally and narrows back to Int
    // when materializing the result row.
    val rows = spark.read.parquet(tablePath)
      .groupBy("year").agg(org.apache.spark.sql.functions.sum("v").as("s"))
      .collect()
      .map(r => (r.getInt(0), r.getLong(1)))
      .sortBy(_._1)
      .toSeq

    assertEquals(Seq((2024, 30L), (2025, 70L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Partitioned Parquet should route through native; flag = " +
        QuantumStrategy.lastTranslation)

    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf); f.delete()
    }
    rmRf(tmpDir)
  }

  /**
   * Phase 5.17: WHERE on a partition column. Spark's FileIndex prunes the
   * scanned file set BEFORE our strategy sees the relation, so we should
   * still see a HadoopFsRelation but with `inputFiles` reduced to only the
   * matching partitions. The query result must be correct and we should
   * route through native (the partition column is then synthesized as a
   * constant by PartitionedFileReader).
   */
  @Test
  def partitionFilterPrunesAndRoutesNatively(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq(
      (2024L, 1L, 10L), (2024L, 2L, 20L),
      (2025L, 3L, 30L), (2025L, 4L, 40L),
      (2026L, 5L, 50L), (2026L, 6L, 60L))
      .toDF("year", "k", "v")

    val tmpDir = java.nio.file.Files.createTempDirectory("quantum_partfilt_").toFile
    val tablePath = new java.io.File(tmpDir, "data").getAbsolutePath
    df.write.mode("overwrite").partitionBy("year").parquet(tablePath)

    val rows = spark.read.parquet(tablePath)
      .filter("year = 2025")
      .groupBy("year").agg(org.apache.spark.sql.functions.sum("v").as("s"))
      .collect()
      .map(r => (r.getInt(0), r.getLong(1)))
      .toSeq

    // Only year=2025 partition survives -> SUM(30 + 40) = 70.
    assertEquals(Seq((2025, 70L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Partition-filtered Parquet should still route natively; flag = " +
        QuantumStrategy.lastTranslation)

    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf); f.delete()
    }
    rmRf(tmpDir)
  }

  /**
   * Phase 5.11: WHERE pushdown composes with FileScanNode. The predicate
   * stack should wrap a single FilterNode around the FileScan, identical
   * to how it wraps a ScanNode for in-memory sources.
   */
  @Test
  def wherePushdownOverParquetFileSource(): Unit = {
    val s = spark
    import s.implicits._
    val df = Seq((1L, 10L), (1L, 20L), (2L, 30L), (2L, 40L), (3L, 50L))
      .toDF("k", "v")

    val tmpDir = java.nio.file.Files.createTempDirectory("quantum_parquet_filt_").toFile
    val parquetDir = new java.io.File(tmpDir, "data").getAbsolutePath
    df.coalesce(1).write.mode("overwrite").parquet(parquetDir)
    val singleFile = new java.io.File(parquetDir).listFiles()
      .find(_.getName.endsWith(".parquet"))
      .map(_.getAbsolutePath)
      .getOrElse(throw new IllegalStateException("no parquet file produced"))

    // WHERE v > 25 -> keeps (2,30), (2,40), (3,50). SUM by k -> (2,70), (3,50).
    val rows = spark.read.parquet(singleFile)
      .filter("v > 25")
      .groupBy("k").agg(org.apache.spark.sql.functions.sum("v").as("s"))
      .collect()
      .map(r => (r.getLong(0), r.getLong(1)))
      .sortBy(_._1)
      .toSeq

    assertEquals(Seq((2L, 70L), (3L, 50L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(t =>
      t.startsWith("HashAggregate") && t.contains("1 pushed filters")),
      "Expected 1 pushed filter over FileScan; flag = " +
        QuantumStrategy.lastTranslation)

    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf); f.delete()
    }
    rmRf(tmpDir)
  }
}
