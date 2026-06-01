package quantum.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, GenericInternalRow, UnsafeProjection}
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StringType}
import org.apache.spark.unsafe.types.UTF8String
import quantum.QuantumEngine

/**
 * One column of input data being handed to the native engine. The native
 * side has a typed column builder per dataType, so we discriminate on the
 * Scala side and dispatch to the right `q_column_*` builder.
 */
sealed trait InputColumn
/**
 * @param values   row values (slot value is irrelevant when validity bit is 0)
 * @param validity null = all-valid; otherwise per-row 0/1 bytes — must match
 *                 `values.length`
 */
final case class LongColumn(values: Array[Long], validity: Array[Byte] = null) extends InputColumn
final case class StringColumn(values: Array[String])                           extends InputColumn

/**
 * Custom SparkPlan: hands an already-translated PlanNode IR over to the
 * native Quantum Engine via the JNA bridge, returns the results as an RDD
 * of InternalRow back to Spark.
 *
 * Phase 5.7 scope: input columns may be Long or String (UTF-8). Output
 * columns may be Long, Double, or String — dispatch in both directions
 * is by declared dataType (input via `inputColumns: InputColumn`, output
 * via `output: Seq[Attribute]`).
 *
 * The native engine emits result columns in the order
 * `[group keys in plan order, aggregates in plan order]`, which doesn't
 * have to match Spark's expected output ordering — `physicalIndex(c)`
 * tells us which native column to pull for Spark output column `c`.
 */
case class QuantumExec(
    override val output: Seq[Attribute],
    inputColumns: Array[InputColumn],
    sourceId: String,
    planBytes: Array[Byte],
    physicalIndex: Array[Int])
  extends LeafExecNode {

  override protected def doExecute(): RDD[InternalRow] = {
    val rows = runNative()
    val proj = UnsafeProjection.create(output, output)
    val unsafe = rows.map(r => proj(r).copy())
    sparkContext.parallelize(unsafe, 1)
  }

  private def runNative(): Seq[InternalRow] = {
    val eng = QuantumEngine.createEngine(64L * 1024 * 1024)
    try {
      // File-source plans need no source registration — the FileScanNode in
      // the plan tells the engine to open the file itself. In-memory plans
      // pack the data into a single Batch and register it under sourceId.
      if (inputColumns.nonEmpty) {
        val cols: Array[QuantumEngine.Column] = inputColumns.map {
          case LongColumn(v, va) => QuantumEngine.columnInt64(eng, v, va)
          case StringColumn(v)   => QuantumEngine.columnString(eng, v)
        }
        val batch = QuantumEngine.batchNew(eng, cols)
        QuantumEngine.registerSource(eng, sourceId, Array(batch))
      }

      val result = QuantumEngine.execute(eng, planBytes)
      try {
        val collected = scala.collection.mutable.ArrayBuffer.empty[InternalRow]
        val nOut      = output.size
        var out: QuantumEngine.Batch = null
        while ({ out = QuantumEngine.resultNext(result); out != null }) {
          val nRows = QuantumEngine.batchNumRows(out)

          // Pre-fetch column data once per batch where it's a contiguous
          // primitive array. For STRING we read row-by-row inside the loop
          // (no contiguous representation across the JNA boundary).
          val colData = new Array[AnyRef](nOut)
          var c = 0
          while (c < nOut) {
            val physCol = physicalIndex(c).toLong
            colData(c) = output(c).dataType match {
              case LongType    => QuantumEngine.batchColInt64(out, physCol, nRows)
              // Int outputs are stored as INT64 natively; narrow at row time.
              case IntegerType => QuantumEngine.batchColInt64(out, physCol, nRows)
              case DoubleType  => QuantumEngine.batchColFloat64(out, physCol, nRows)
              case StringType  => null  // sentinel: read per-row via batchColStringAt
              case other       =>
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
                values(ci) = output(ci).dataType match {
                  case LongType =>
                    (colData(ci).asInstanceOf[Array[Long]](r.toInt): java.lang.Long)
                  case IntegerType =>
                    (colData(ci).asInstanceOf[Array[Long]](r.toInt).toInt: java.lang.Integer)
                  case DoubleType =>
                    (colData(ci).asInstanceOf[Array[Double]](r.toInt): java.lang.Double)
                  case StringType =>
                    val s = QuantumEngine.batchColStringAt(out, physCol, r.toLong)
                    if (s == null) null else UTF8String.fromString(s)
                  case other =>
                    throw new RuntimeException(s"QuantumExec: bad output type $other")
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
