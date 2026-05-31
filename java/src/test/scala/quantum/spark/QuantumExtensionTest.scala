package quantum.spark

import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.Test

/**
 * Confirms that QuantumExtension is loaded into a SparkSession and that the
 * planner strategy fires on every query. Full plan rewriting + native
 * execution is the next slice.
 */
class QuantumExtensionTest {

  @Test
  def extensionStrategyFiresOnSqlQuery(): Unit = {
    QuantumStrategy.reset()

    val spark = SparkSession.builder()
      .appName("quantum-extension-test")
      .master("local[1]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "1")
      .withExtensions(new QuantumExtension())
      .getOrCreate()

    try {
      // A trivial query is enough to exercise the planner; the strategy
      // records the LogicalPlan it sees into a thread-local.
      val df = spark.sql("SELECT id, id + 1 AS plus1 FROM range(5) WHERE id > 1")
      val rows = df.collect().map(r => (r.getLong(0), r.getLong(1))).toSeq

      assertEquals(Seq((2L, 3L), (3L, 4L), (4L, 5L)), rows)

      val observed = QuantumStrategy.lastObservedPlan
      assertTrue(observed.isDefined,
                 "QuantumStrategy should have observed at least one LogicalPlan")
    } finally {
      spark.stop()
    }
  }
}
