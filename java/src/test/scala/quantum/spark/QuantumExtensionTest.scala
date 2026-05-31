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
}
