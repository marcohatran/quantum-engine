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
      // AQE wraps subtrees in LogicalQueryStage, which hides the Aggregate
      // from our planner strategy. Disable for this slice; AQE-friendly
      // pattern matching is a follow-up.
      .config("spark.sql.adaptive.enabled", "false")
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
}
