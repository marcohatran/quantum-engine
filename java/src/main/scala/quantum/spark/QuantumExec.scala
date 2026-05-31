package quantum.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, GenericInternalRow, UnsafeProjection}
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.types.{DoubleType, LongType}
import quantum.QuantumEngine

/**
 * Custom SparkPlan: hands an already-translated PlanNode IR over to the
 * native Quantum Engine via the JNA bridge, returns the results as an RDD
 * of InternalRow back to Spark.
 *
 * Phase 5.5 scope: the input is a pre-extracted column-major dataset of
 * Long values. The plan is a HashAggregate with INT64 group keys and any
 * combination of SUM(int64), MIN(int64), MAX(int64), AVG(int64), and
 * COUNT(*) aggregates. AVG produces a Double output column; everything
 * else is Long.
 *
 * The native engine emits result columns in the order
 * `[group keys in plan order, aggregates in plan order]`, which doesn't
 * have to match Spark's expected output ordering. The strategy passes a
 * `physicalIndex` array that tells us, for each Spark output column,
 * which native column to pull from.
 *
 * @param output         the projected output attributes (Spark's expected
 *                       ordering — drives type dispatch when reading
 *                       columns out of the result batches)
 * @param inputColumns   column-major Long data — first dim is column
 *                       index in the LocalRelation source, second is row
 * @param sourceId       protobuf scan source_id under which the input
 *                       columns are registered
 * @param planBytes      serialized PlanNode for the native engine
 * @param physicalIndex  output(c) is read from native column physicalIndex(c)
 */
case class QuantumExec(
    override val output: Seq[Attribute],
    inputColumns: Array[Array[Long]],
    sourceId: String,
    planBytes: Array[Byte],
    physicalIndex: Array[Int])
  extends LeafExecNode {

  override protected def doExecute(): RDD[InternalRow] = {
    val rows = runNative()
    // Spark's downstream paths (collect / shuffle / codegen) expect UnsafeRow.
    // Project our GenericInternalRow rows through UnsafeProjection now so the
    // RDD carries UnsafeRow elements.
    val proj = UnsafeProjection.create(output, output)
    val unsafe = rows.map(r => proj(r).copy())
    sparkContext.parallelize(unsafe, 1)
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
        val nOut      = output.size
        var out: QuantumEngine.Batch = null
        while ({ out = QuantumEngine.resultNext(result); out != null }) {
          val nRows = QuantumEngine.batchNumRows(out)

          // Pre-fetch each Spark-output column from its native source column,
          // dispatching on declared dataType.
          val colData = new Array[AnyRef](nOut)
          var c = 0
          while (c < nOut) {
            val physCol = physicalIndex(c).toLong
            colData(c) = output(c).dataType match {
              case LongType   => QuantumEngine.batchColInt64(out, physCol, nRows)
              case DoubleType => QuantumEngine.batchColFloat64(out, physCol, nRows)
              case other      =>
                throw new RuntimeException(
                  s"QuantumExec: unsupported output type $other for column ${output(c).name}")
            }
            c += 1
          }

          var r = 0
          while (r < nRows) {
            val values = new Array[Any](nOut)
            var ci = 0
            while (ci < nOut) {
              val physCol = physicalIndex(ci).toLong
              val isValid = QuantumEngine.batchColValid(out, physCol, r.toLong)
              if (output(ci).nullable && !isValid) {
                values(ci) = null
              } else {
                values(ci) = colData(ci) match {
                  case arr: Array[Long]   => (arr(r.toInt): java.lang.Long)
                  case arr: Array[Double] => (arr(r.toInt): java.lang.Double)
                  case other =>
                    throw new RuntimeException(s"QuantumExec: bad colData $other")
                }
              }
              ci += 1
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
