package quantum.spark

import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}

/**
 * Spark → Delta end-to-end. Delta-spark exposes Delta tables as a
 * `LogicalRelation(HadoopFsRelation(TahoeLogFileIndex, ..., ParquetFileFormat))`,
 * which means our existing peel match for Parquet HadoopFsRelations (added in
 * Phase 5.9 / 5.10) should already route them — Spark sees a Delta table as
 * "the set of currently-active Parquet part files," exactly the shape we
 * already handle via ConcatFileReader.
 *
 * If this premise holds, no strategy code change is needed for Delta; we
 * just need to verify it works.
 */
class QuantumDeltaTest {

  private var spark: SparkSession = _

  @BeforeEach
  def setUp(): Unit = {
    QuantumStrategy.reset()
    spark = SparkSession.builder()
      .appName("quantum-delta-test")
      .master("local[1]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "1")
      // Stack Delta's session extension under ours via the config-based
      // path; SparkSessionExtensions accumulates across config and
      // withExtensions, so both Delta and Quantum register their rules.
      .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
      .config("spark.sql.catalog.spark_catalog",
              "org.apache.spark.sql.delta.catalog.DeltaCatalog")
      .config("spark.sql.optimizer.excludedRules",
              "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation")
      .withExtensions(new QuantumExtension())
      .getOrCreate()
  }

  @AfterEach
  def tearDown(): Unit = if (spark != null) spark.stop()

  @Test
  def deltaTableReadsThroughNativeFileScan(): Unit = {
    val s = spark
    import s.implicits._

    val df = Seq((1L, 10L), (1L, 20L), (2L, 30L), (2L, 40L), (3L, 50L))
      .toDF("k", "v")

    val tmpDir = java.nio.file.Files.createTempDirectory("quantum_delta_").toFile
    val deltaDir = new java.io.File(tmpDir, "data").getAbsolutePath
    df.coalesce(1).write.format("delta").mode("overwrite").save(deltaDir)

    // Sanity: the Delta log was written.
    val deltaLog = new java.io.File(deltaDir, "_delta_log")
    assertTrue(deltaLog.isDirectory, s"expected Delta log at $deltaLog")

    val rows = spark.read.format("delta").load(deltaDir)
      .groupBy("k").agg(org.apache.spark.sql.functions.sum("v").as("s"))
      .collect()
      .map(r => (r.getLong(0), r.getLong(1)))
      .sortBy(_._1)
      .toSeq

    assertEquals(Seq((1L, 30L), (2L, 70L), (3L, 50L)), rows)
    assertTrue(QuantumStrategy.lastTranslation.exists(_.startsWith("HashAggregate")),
      "Delta table should route through native; flag = " +
        QuantumStrategy.lastTranslation +
        "; observed plan classes = " + QuantumStrategy.observedClasses)

    def rmRf(f: java.io.File): Unit = {
      if (f.isDirectory) f.listFiles().foreach(rmRf); f.delete()
    }
    rmRf(tmpDir)
  }
}
