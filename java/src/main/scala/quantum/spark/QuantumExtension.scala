package quantum.spark

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeReference, EqualTo, ExprId, Expression, GreaterThan, GreaterThanOrEqual, IsNotNull, LessThan, LessThanOrEqual, Literal, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, AggregateFunction, Average, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, LocalRelation, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.types.{DataType, IntegerType, LongType, StringType}
import quantum.plan.proto.QuantumPlan

import java.util.concurrent.atomic.AtomicReference

/**
 * Spark SessionExtensions hook for Quantum Engine.
 *
 * Wire it into a SparkSession via:
 * {{{
 *   SparkSession.builder()
 *     .withExtensions(new QuantumExtension())
 *     .getOrCreate()
 * }}}
 *
 * Or via Spark conf: `spark.sql.extensions=quantum.spark.QuantumExtension`.
 *
 * Recognises a small set of LogicalPlan shapes and translates them to the
 * engine's protobuf PlanNode IR for native execution. Anything unrecognised
 * falls through to Spark's default planner.
 */
class QuantumExtension extends (SparkSessionExtensions => Unit) {
  override def apply(ext: SparkSessionExtensions): Unit = {
    ext.injectPlannerStrategy(_ => new QuantumStrategy())
  }
}

class QuantumStrategy extends SparkStrategy {

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = {
    QuantumStrategy.lastSeen.set(plan)
    QuantumStrategy.seenClasses.add(plan.getClass.getSimpleName)
    plan match {
      case agg @ Aggregate(groupingExprs, aggregateExprs, child) =>
        QuantumStrategy.tryTranslateLongAggregate(agg, groupingExprs, aggregateExprs, child)
          .map(quantumExec => Seq(quantumExec))
          .getOrElse(Nil)
      case _ => Nil
    }
  }
}

/**
 * Where the rows for the engine's scan come from. Either Spark already
 * materialized them as a LocalRelation (we marshal column-major into the
 * engine via JNA) or they live on disk and the engine reads them itself via
 * a `FileScanNode` (Phase 5.9+).
 */
sealed trait DataSource {
  def output: Seq[Attribute]
}
final case class InMemorySource(lr: LocalRelation) extends DataSource {
  override def output: Seq[Attribute] = lr.output
}
final case class FileSource(
    paths: Seq[String],
    format: String,
    override val output: Seq[Attribute]) extends DataSource

object QuantumStrategy {
  // Thread-local sinks for tests: the most-recent LogicalPlan considered, and
  // a label set when we actually translated something to native.
  val lastSeen: AtomicReference[LogicalPlan] = new AtomicReference[LogicalPlan]()
  val lastTranslated: AtomicReference[String] = new AtomicReference[String]()
  val seenClasses: java.util.Set[String] =
    java.util.Collections.synchronizedSet(new java.util.LinkedHashSet[String]())

  def reset(): Unit = {
    lastSeen.set(null)
    lastTranslated.set(null)
    seenClasses.clear()
  }
  def lastObservedPlan: Option[LogicalPlan] = Option(lastSeen.get())
  def lastTranslation: Option[String] = Option(lastTranslated.get())
  def observedClasses: Set[String] = {
    import scala.jdk.CollectionConverters._
    seenClasses.asScala.toSet
  }

  /**
   * Phase 5.5 scope: recognise `Aggregate(groupingExprs, aggregateExprs,
   * <peelable chain over LocalRelation of all-Long columns>)` where:
   *
   *  - The peeled chain (SubqueryAlias / column-renaming Project) ends in
   *    a LocalRelation whose every column is LongType.
   *  - Each grouping expression is an AttributeReference that refers to a
   *    LocalRelation column.
   *  - Each aggregateExpr is either a bare reference to a grouping key, or
   *    an AggregateExpression wrapping one of:
   *      SUM(long), MIN(long), MAX(long), AVG(long), COUNT(*).
   *
   * Everything else falls through to Spark's default planner.
   */
  def tryTranslateLongAggregate(
      agg: Aggregate,
      groupingExprs: Seq[Expression],
      aggregateExprs: Seq[NamedExpression],
      child: LogicalPlan): Option[SparkPlan] = {

    // --- Predicate translator parameterised on the exprId-to-LR-column map
    //     valid at the level it's called. Filters peeled at different depths
    //     of the chain use different maps; we translate each one against the
    //     map that was valid at the Filter's depth so the resulting ScalarExpr
    //     is keyed on LR column indices (depth-independent).
    def flattenAnd(e: Expression): Seq[Expression] = e match {
      case And(l, r) => flattenAnd(l) ++ flattenAnd(r)
      case other     => Seq(other)
    }

    // When Spark reads from a file source, it wraps user predicates like
    // `v > 25` in `isnotnull(v) AND v > 25` — the IsNotNull clause helps
    // parquet skip NULL pages. The native engine's comparison ops already
    // null-propagate (NULL inputs -> NULL output -> excluded by Filter),
    // so an IsNotNull(col) conjunct is redundant whenever the same column
    // appears in any comparison conjunct. Drop those.
    def stripRedundantIsNotNull(conjuncts: Seq[Expression]): Seq[Expression] = {
      def comparedCols(e: Expression): Seq[ExprId] = e match {
        case GreaterThan(l, r) => comparedCols(l) ++ comparedCols(r)
        case LessThan(l, r)    => comparedCols(l) ++ comparedCols(r)
        case EqualTo(l, r)     => comparedCols(l) ++ comparedCols(r)
        case ar: AttributeReference => Seq(ar.exprId)
        case _ => Seq.empty
      }
      val compared: Set[ExprId] = conjuncts.flatMap(comparedCols).toSet
      conjuncts.filter {
        case IsNotNull(ar: AttributeReference) => !compared.contains(ar.exprId)
        case _ => true
      }
    }

    def translateScalarWith(idMap: Map[ExprId, Int])(e: Expression): Option[QuantumPlan.ScalarExpr] = {
      def binOp(op: QuantumPlan.BinaryOpType, l: Expression, r: Expression) =
        for {
          ls <- translateScalarWith(idMap)(l)
          rs <- translateScalarWith(idMap)(r)
        } yield QuantumPlan.ScalarExpr.newBuilder()
          .setBinary(QuantumPlan.BinaryOp.newBuilder()
            .setOp(op).setLhs(ls).setRhs(rs))
          .build()

      e match {
        case ar: AttributeReference =>
          idMap.get(ar.exprId).map { idx =>
            QuantumPlan.ScalarExpr.newBuilder()
              .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(idx))
              .build()
          }
        case Literal(v: java.lang.Long, LongType) =>
          Some(QuantumPlan.ScalarExpr.newBuilder()
            .setLiteral(QuantumPlan.Literal.newBuilder().setInt64Value(v))
            .build())
        case Literal(v: java.lang.Integer, IntegerType) =>
          Some(QuantumPlan.ScalarExpr.newBuilder()
            .setLiteral(QuantumPlan.Literal.newBuilder().setInt64Value(v.longValue))
            .build())
        case GreaterThan(l, r) => binOp(QuantumPlan.BinaryOpType.BINOP_GT, l, r)
        case LessThan(l, r)    => binOp(QuantumPlan.BinaryOpType.BINOP_LT, l, r)
        case EqualTo(l, r)     => binOp(QuantumPlan.BinaryOpType.BINOP_EQ, l, r)
        case _: GreaterThanOrEqual | _: LessThanOrEqual => None
        case _ => None
      }
    }

    // Peel collects already-translated FilterNode predicates so each one is
    // bound to the column map that was valid at the Filter's depth. The leaf
    // is either a LocalRelation (in-memory inputs) or a Parquet-backed
    // LogicalRelation (file scan — engine reads the file itself).
    def peel(p: LogicalPlan): Option[(DataSource, Map[ExprId, Int], Seq[QuantumPlan.ScalarExpr])] =
      p match {
        case lr: LocalRelation =>
          val m = lr.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
          Some((InMemorySource(lr), m, Seq.empty))
        case LogicalRelation(hr: HadoopFsRelation, output, _, _) =>
          hr.fileFormat match {
            case _: ParquetFileFormat =>
              // The native ConcatFileReader handles N>=1 part files; the
              // strategy just needs to hand them all over in stable order.
              val paths = hr.location.inputFiles.toSeq.sorted
              if (paths.isEmpty) {
                lastTranslated.set("reject: Parquet LogicalRelation has no files")
                None
              } else {
                val m = output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
                Some((FileSource(paths, "parquet", output), m, Seq.empty))
              }
            case other =>
              lastTranslated.set(s"reject: unsupported HadoopFsRelation format $other")
              None
          }
        case sa: SubqueryAlias =>
          peel(sa.child)
        case f: Filter =>
          peel(f.child).flatMap { case (src, m, accum) =>
            val conjs      = stripRedundantIsNotNull(flattenAnd(f.condition))
            val translated = conjs.map(translateScalarWith(m))
            if (translated.exists(_.isEmpty)) {
              lastTranslated.set(s"reject: predicate '${f.condition}' untranslatable " +
                "(only Long >, <, = literal/col supported)")
              None
            } else {
              Some((src, m, accum ++ translated.flatten))
            }
          }
        case pr: Project =>
          peel(pr.child).flatMap { case (src, childMap, accum) =>
            val pairs: Seq[Option[(ExprId, Int)]] = pr.projectList.zip(pr.output).map {
              case (ar: AttributeReference, outAttr) =>
                childMap.get(ar.exprId).map(outAttr.exprId -> _)
              case (Alias(ar: AttributeReference, _), outAttr) =>
                childMap.get(ar.exprId).map(outAttr.exprId -> _)
              case _ => None
            }
            if (pairs.exists(_.isEmpty)) None
            else Some((src, pairs.flatten.toMap, accum))
          }
        case _ => None
      }

    def isSupportedColType(t: DataType): Boolean = t match {
      case LongType | StringType => true
      case _                     => false
    }

    val (dataSource, exprIdToCol, pushedFilters) = peel(child) match {
      case Some(triple) if triple._1.output.forall(a => isSupportedColType(a.dataType)) => triple
      case Some((src, _, _)) =>
        lastTranslated.set(s"reject: data source has unsupported column types (" +
          src.output.map(a => s"${a.name}:${a.dataType}").mkString(", ") + ")")
        return None
      case None =>
        if (lastTranslated.get == null || !lastTranslated.get.startsWith("reject:"))
          lastTranslated.set("reject: child didn't peel to a known source\n" + child.treeString)
        return None
    }

    // --- Grouping: each grouping expr must be an AttributeReference resolvable
    //     to a LocalRelation column index. (Type is whatever the LocalRelation
    //     column declared; the native HashAggregate decides which path to use.)
    val groupColIndices: Seq[Int] = groupingExprs.flatMap {
      case ar: AttributeReference => exprIdToCol.get(ar.exprId)
      case _                      => None
    }
    if (groupColIndices.size != groupingExprs.size) {
      lastTranslated.set("reject: grouping exprs include non-AttributeReference or unresolved ref")
      return None
    }
    val groupKeySet: Set[ExprId] = groupingExprs.collect {
      case ar: AttributeReference => ar.exprId
    }.toSet

    // --- Output projection: each output expr is either a group-key reference
    //     or an AggregateExpression we know how to translate.
    sealed trait OutputItem
    case class GroupKeyOut(colIdx: Int)                          extends OutputItem
    case class AggOut(spec: QuantumPlan.AggregateSpec, label: String) extends OutputItem

    def aggCol(colIdx: Int): QuantumPlan.ScalarExpr =
      QuantumPlan.ScalarExpr.newBuilder()
        .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(colIdx))
        .build()

    def specOf(kind: QuantumPlan.AggregateKind, inputCol: Option[Int]): QuantumPlan.AggregateSpec = {
      val b = QuantumPlan.AggregateSpec.newBuilder()
        .setKind(kind)
        .setInputType(QuantumPlan.TypeId.TYPE_INT64)
      inputCol.foreach(idx => b.setInput(aggCol(idx)))
      b.build()
    }

    def classifyAgg(fn: AggregateFunction): Option[OutputItem] = fn match {
      case Sum(ar: AttributeReference, _) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(idx => AggOut(specOf(QuantumPlan.AggregateKind.AGG_SUM, Some(idx)), "sum"))
      case Min(ar: AttributeReference) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(idx => AggOut(specOf(QuantumPlan.AggregateKind.AGG_MIN, Some(idx)), "min"))
      case Max(ar: AttributeReference) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(idx => AggOut(specOf(QuantumPlan.AggregateKind.AGG_MAX, Some(idx)), "max"))
      case Average(ar: AttributeReference, _) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(idx => AggOut(specOf(QuantumPlan.AggregateKind.AGG_AVG, Some(idx)), "avg"))
      case Count(children) if children.forall {
        case _: Literal => true
        case _          => false
      } =>
        // COUNT(*) is parsed as Count(Seq(Literal(1, IntegerType))). We only
        // support COUNT(*) for now; COUNT(<column>) would need a separate
        // AGG_COUNT_COLUMN kind in the native engine.
        Some(AggOut(specOf(QuantumPlan.AggregateKind.AGG_COUNT_STAR, None), "count_star"))
      case _ => None
    }

    val outputItems: Seq[OutputItem] = aggregateExprs.flatMap {
      // Bare or aliased grouping-key reference.
      case ar: AttributeReference if groupKeySet.contains(ar.exprId) =>
        exprIdToCol.get(ar.exprId).map(GroupKeyOut)
      case Alias(ar: AttributeReference, _) if groupKeySet.contains(ar.exprId) =>
        exprIdToCol.get(ar.exprId).map(GroupKeyOut)
      // Aggregate. In Aggregate.aggregateExpressions the AggregateExpression
      // is always wrapped in an Alias (NamedExpression requirement).
      case Alias(AggregateExpression(fn, _, _, _, _), _) => classifyAgg(fn)
      case other =>
        lastTranslated.set(s"reject: unsupported output expression: $other (${other.getClass.getSimpleName})")
        return None
    }
    if (outputItems.size != aggregateExprs.size) {
      lastTranslated.set("reject: some output expressions couldn't be classified (unsupported agg or unresolved col)")
      return None
    }

    // --- Build the proto plan. For an in-memory source the scan is a
    //     ScanNode that references an engine-side registered source. For a
    //     file source the scan is a FileScanNode and no data marshalling
    //     happens — the engine opens the file itself via io::FileSystem.
    def protoTypeFor(t: DataType): QuantumPlan.TypeId = t match {
      case LongType   => QuantumPlan.TypeId.TYPE_INT64
      case StringType => QuantumPlan.TypeId.TYPE_STRING
      case other      => throw new IllegalStateException(s"unsupported source col type $other")
    }
    val (sourceId, scan) = dataSource match {
      case InMemorySource(_) =>
        val schemaB = QuantumPlan.Schema.newBuilder()
        dataSource.output.zipWithIndex.foreach { case (a, i) =>
          schemaB.addFields(QuantumPlan.Field.newBuilder()
            .setName(if (a.name.nonEmpty) a.name else s"c$i")
            .setType(protoTypeFor(a.dataType)))
        }
        val sid = s"quantum_src_${System.nanoTime()}"
        val s = QuantumPlan.PlanNode.newBuilder()
          .setScan(QuantumPlan.ScanNode.newBuilder().setSourceId(sid).setSchema(schemaB))
          .build()
        (sid, s)
      case FileSource(paths, format, _) =>
        val fs = QuantumPlan.FileScanNode.newBuilder().setFormat(format)
        paths.foreach(fs.addPaths)
        // FileScanNode.schema is left unset — the reader derives it from
        // the file's own metadata.
        val s = QuantumPlan.PlanNode.newBuilder().setFileScan(fs).build()
        ("", s)
    }

    // --- Wrap the Scan in one FilterNode per pre-translated pushed-down
    //     predicate. The native BinaryOp has no AND, so flattenAnd already
    //     decomposed conjunctions when each Filter was peeled.
    val scanWithFilters: QuantumPlan.PlanNode = pushedFilters.foldLeft(scan) { (acc, pred) =>
      QuantumPlan.PlanNode.newBuilder()
        .setFilter(QuantumPlan.FilterNode.newBuilder()
          .setChild(acc)
          .setPredicate(pred))
        .build()
    }

    val hashAggB = QuantumPlan.HashAggregateNode.newBuilder().setChild(scanWithFilters)
    groupColIndices.foreach(idx => hashAggB.addGroupKeys(aggCol(idx)))

    // Project output expressions onto either group-key positions (output names
    // are taken from the Aggregate's output schema, which Catalyst already
    // disambiguated) or aggregate spec slots.
    val outputNames = agg.output.map(_.name)
    // The native operator emits: [group keys in declared order] ++ [aggregates in declared order].
    // Spark's `aggregateExprs` mixes them, so we have to reorder.
    val orderedAggs: Seq[QuantumPlan.AggregateSpec] = outputItems.collect {
      case AggOut(spec, _) => spec
    }
    orderedAggs.foreach(hashAggB.addAggregates)
    // Output names: native emits group-key columns first (in groupingExprs
    // order), then aggregate columns (in declaration order). Reassemble names
    // in that physical order so the JVM-side mapping reads them correctly.
    val groupNamesInOrder: Seq[String] = groupingExprs.zipWithIndex.map { case (_, i) =>
      // For Phase 5.5 we always reproject these into `agg.output` order on the
      // JVM side anyway, so the names here are just for the engine's IR debug.
      s"g$i"
    }
    val aggNamesInOrder: Seq[String] = outputItems.collect { case AggOut(_, label) => label }
    (groupNamesInOrder ++ aggNamesInOrder).foreach(hashAggB.addOutputNames)

    val root = QuantumPlan.PlanNode.newBuilder().setHashAgg(hashAggB).build()

    // --- Determine physical-to-Spark output reordering.
    // Native column order = (group keys in groupingExprs order, then aggs in
    // outputItems-AggOut order). Spark wants `aggregateExprs` order.
    val physicalIndexBuilder = scala.collection.mutable.ArrayBuffer.empty[Int]
    val nGroupKeys = groupingExprs.size
    var aggSlot    = 0
    var groupSlot  = 0
    // Build a map from groupingExpr position -> physical native column.
    val groupExprIdToPhysicalIdx: Map[ExprId, Int] = groupingExprs.zipWithIndex.collect {
      case (ar: AttributeReference, i) => ar.exprId -> i
    }.toMap

    outputItems.foreach {
      case GroupKeyOut(localColIdx) =>
        // Find which groupingExpr position references this localColIdx and
        // map to that physical native index.
        val phys = groupingExprs.zipWithIndex.collectFirst {
          case (ar: AttributeReference, i) if exprIdToCol.get(ar.exprId).contains(localColIdx) => i
        }.getOrElse {
          throw new IllegalStateException("GroupKeyOut without matching groupingExpr")
        }
        physicalIndexBuilder += phys
      case AggOut(_, _) =>
        physicalIndexBuilder += (nGroupKeys + aggSlot)
        aggSlot += 1
    }
    val physicalIndex: Array[Int] = physicalIndexBuilder.toArray
    val _ = (outputNames, groupSlot, groupExprIdToPhysicalIdx) // silence unused

    // --- Marshal source rows. For an InMemorySource we copy the
    //     LocalRelation's row data into column-major typed arrays. For a
    //     FileSource we hand off an empty array; the FileScanNode tells the
    //     native engine to open the file itself.
    val inputCols: Array[InputColumn] = dataSource match {
      case InMemorySource(lr) =>
        val data   = lr.data.toArray
        val nCols  = lr.output.size
        Array.tabulate(nCols) { c =>
          lr.output(c).dataType match {
            case LongType =>
              LongColumn(data.map(_.getLong(c)))
            case StringType =>
              StringColumn(data.map { row =>
                if (row.isNullAt(c)) null else row.getString(c)
              })
            case other =>
              throw new IllegalStateException(s"unsupported LR col type $other")
          }
        }
      case _: FileSource =>
        Array.empty[InputColumn]
    }

    lastTranslated.set("HashAggregate(" + aggregateExprs.size + " outputs over " +
      groupColIndices.size + " group keys, " + pushedFilters.size + " pushed filters)")
    Some(QuantumExec(
      output         = agg.output,
      inputColumns   = inputCols,
      sourceId       = sourceId,
      planBytes      = root.toByteArray,
      physicalIndex  = physicalIndex))
  }
}
