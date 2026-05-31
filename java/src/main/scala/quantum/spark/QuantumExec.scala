package quantum.spark

import com.google.protobuf.ByteString
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, GenericInternalRow}
import org.apache.spark.sql.execution.LeafExecNode
import quantum.QuantumEngine
import quantum.plan.proto.QuantumPlan

import scala.jdk.CollectionConverters._

/**
 * Custom SparkPlan: hands an already-translated PlanNode IR over to the
 * native Quantum Engine via the JNA bridge, returns the results as an RDD
 * of InternalRow back to Spark.
 *
 * Phase 4.7 scope: the input is a pre-extracted column-major dataset of
 * Long values; the plan is a HashAggregate with INT64 keys and SUM(INT64)
 * aggregates. The pattern matcher in QuantumStrategy is responsible for
 * extracting `inputColumns` from the matched LogicalPlan.
 *
 * @param output       the projected output attributes (matches plan output)
 * @param inputColumns column-major Long data — first dim is column index,
 *                     second is row.
 * @param sourceId     the protobuf scan source_id this column data is registered against
 * @param planBytes    serialized PlanNode for the native engine
 */
case class QuantumExec(
    override val output: Seq[Attribute],
    inputColumns: Array[Array[Long]],
    sourceId: String,
    planBytes: Array[Byte])
  extends LeafExecNode {

  override protected def doExecute(): RDD[InternalRow] = {
    val rows = runNative()
    sparkContext.parallelize(rows, 1)
  }

  private def runNative(): Seq[InternalRow] = {
    val eng = QuantumEngine.createEngine(64L * 1024 * 1024)
    try {
      val cols = inputColumns.map(c =>
        QuantumEngine.columnInt64(eng, c, null))
      val batch = QuantumEngine.batchNew(eng, cols)
      QuantumEngine.registerSource(eng, sourceId, Array(batch))

      val result = QuantumEngine.execute(eng, planBytes)
      try {
        val collected = scala.collection.mutable.ArrayBuffer.empty[InternalRow]
        var out: QuantumEngine.Batch = null
        while ({ out = QuantumEngine.resultNext(result); out != null }) {
          val nRows = QuantumEngine.batchNumRows(out)
          val nCols = QuantumEngine.batchNumCols(out).toInt
          val colData = (0 until nCols).map(c =>
            QuantumEngine.batchColInt64(out, c.toLong, nRows))
          var r = 0
          while (r < nRows) {
            val values = new Array[Any](nCols)
            var c = 0
            while (c < nCols) {
              values(c) = colData(c)(r): java.lang.Long
              c += 1
            }
            collected += new GenericInternalRow(values)
            r += 1
          }
        }
        collected.toSeq
      } finally {
        QuantumEngine.destroyResult(result)
      }
    } finally {
      QuantumEngine.destroyEngine(eng)
    }
  }
}
