package quantum.spark

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.nio.channels.Channels
import java.nio.charset.StandardCharsets

import scala.jdk.CollectionConverters._

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.{BigIntVector, Float8Vector, VarCharVector, VectorSchemaRoot}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema => ArrowSchema}
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
final case class LongColumn(values: Array[Long], validity: Array[Byte] = null)     extends InputColumn
final case class DoubleColumn(values: Array[Double], validity: Array[Byte] = null) extends InputColumn
final case class StringColumn(values: Array[String])                               extends InputColumn

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
/**
 * @param sources       in-memory sources to register before executing the
 *                      plan. Each entry is `(sourceId, columns)`. The
 *                      planBytes' ScanNodes look up these sourceIds. For
 *                      a pure FileScan plan the seq may be empty.
 */
case class QuantumExec(
    override val output: Seq[Attribute],
    sources: Seq[(String, Array[InputColumn])],
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
    val eng       = QuantumEngine.createEngine(64L * 1024 * 1024)
    val allocator = new RootAllocator()
    try {
      // Each registered source (one per LocalRelation in the plan) is
      // serialized as one Arrow IPC stream — a single JNA crossing per
      // source instead of per column. File-source plans (Parquet / Delta /
      // partitioned) have an empty `sources` list — the FileScanNode in
      // the plan handles its own I/O.
      sources.foreach { case (sourceId, cols) =>
        QuantumEngine.registerSourceIpc(eng, sourceId, ipcBytesFor(cols, allocator))
      }

      val result = QuantumEngine.execute(eng, planBytes)
      try {
        val collected = scala.collection.mutable.ArrayBuffer.empty[InternalRow]
        var bytes: Array[Byte] = null
        while ({ bytes = QuantumEngine.resultNextIpc(result); bytes != null }) {
          readIpcBatch(bytes, allocator, collected)
        }
        collected.toSeq
      } finally {
        QuantumEngine.destroyResult(result)
      }
    } finally {
      allocator.close()
      QuantumEngine.destroyEngine(eng)
    }
  }

  /** Serialize one source's columns as an Arrow IPC stream. */
  private def ipcBytesFor(cols: Array[InputColumn],
                          allocator: RootAllocator): Array[Byte] = {
    val fields: Seq[Field] = cols.zipWithIndex.map { case (c, i) =>
      val at: ArrowType = c match {
        case _: LongColumn   => new ArrowType.Int(64, true)
        case _: DoubleColumn => new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
        case _: StringColumn => ArrowType.Utf8.INSTANCE
      }
      new Field(s"c$i", FieldType.nullable(at),
                java.util.Collections.emptyList[Field]())
    }.toSeq
    val nRows = cols.headOption.map {
      case LongColumn(v, _)   => v.length
      case DoubleColumn(v, _) => v.length
      case StringColumn(v)    => v.length
    }.getOrElse(0)

    val root = VectorSchemaRoot.create(new ArrowSchema(fields.asJava), allocator)
    try {
      root.allocateNew()
      cols.zipWithIndex.foreach { case (c, i) =>
        c match {
          case LongColumn(values, validity) =>
            val v = root.getVector(i).asInstanceOf[BigIntVector]
            var r = 0
            while (r < values.length) {
              if (validity == null || validity(r) != 0) v.setSafe(r, values(r))
              else                                      v.setNull(r)
              r += 1
            }
          case DoubleColumn(values, validity) =>
            val v = root.getVector(i).asInstanceOf[Float8Vector]
            var r = 0
            while (r < values.length) {
              if (validity == null || validity(r) != 0) v.setSafe(r, values(r))
              else                                      v.setNull(r)
              r += 1
            }
          case StringColumn(values) =>
            val v = root.getVector(i).asInstanceOf[VarCharVector]
            var r = 0
            while (r < values.length) {
              if (values(r) == null) v.setNull(r)
              else v.setSafe(r, values(r).getBytes(StandardCharsets.UTF_8))
              r += 1
            }
        }
      }
      root.setRowCount(nRows)

      val bos    = new ByteArrayOutputStream()
      val writer = new ArrowStreamWriter(root, null, Channels.newChannel(bos))
      writer.start()
      writer.writeBatch()
      writer.end()
      writer.close()
      bos.toByteArray
    } finally {
      root.close()
    }
  }

  /** Parse one IPC result batch and append its rows (reordered through
   *  physicalIndex) to `out`. Engine columns come back named c0..cN. */
  private def readIpcBatch(bytes: Array[Byte],
                           allocator: RootAllocator,
                           out: scala.collection.mutable.ArrayBuffer[InternalRow]): Unit = {
    val reader = new ArrowStreamReader(new ByteArrayInputStream(bytes), allocator)
    try {
      val root = reader.getVectorSchemaRoot
      val nOut = output.size
      while (reader.loadNextBatch()) {
        val nRows = root.getRowCount
        var r = 0
        while (r < nRows) {
          val values = new Array[Any](nOut)
          var ci = 0
          while (ci < nOut) {
            val vec = root.getVector(physicalIndex(ci))
            if (vec.isNull(r)) {
              values(ci) = null
            } else {
              values(ci) = output(ci).dataType match {
                case LongType =>
                  (vec.asInstanceOf[BigIntVector].get(r): java.lang.Long)
                // Int outputs are widened to INT64 natively; narrow here.
                case IntegerType =>
                  (vec.asInstanceOf[BigIntVector].get(r).toInt: java.lang.Integer)
                case DoubleType =>
                  (vec.asInstanceOf[Float8Vector].get(r): java.lang.Double)
                case StringType =>
                  UTF8String.fromBytes(vec.asInstanceOf[VarCharVector].get(r))
                case other =>
                  throw new RuntimeException(
                    s"QuantumExec: unsupported output type $other for column ${output(ci).name}")
              }
            }
            ci += 1
          }
          out += new GenericInternalRow(values)
          r += 1
        }
      }
    } finally {
      reader.close()
    }
  }
}
