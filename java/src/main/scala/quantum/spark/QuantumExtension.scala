package quantum.spark

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeReference, Contains, EndsWith, EqualTo, ExprId, Expression, GreaterThan, GreaterThanOrEqual, In, IsNotNull, IsNull, LessThan, LessThanOrEqual, Like, Literal, NamedExpression, Not, Or, StartsWith}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, AggregateFunction, Average, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.plans.{FullOuter, Inner, JoinType, LeftOuter, RightOuter}
import org.apache.spark.sql.catalyst.expressions.{Ascending, SortOrder}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, GlobalLimit, Join, LocalLimit, LocalRelation, LogicalPlan, Project, Repartition, RepartitionByExpression, ReturnAnswer, Sort, SubqueryAlias, Union}
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.adaptive.LogicalQueryStage
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.types.{BooleanType, DataType, DoubleType, IntegerType, LongType, StringType}
import org.apache.spark.unsafe.types.UTF8String
import quantum.plan.proto.QuantumPlan

import java.util.concurrent.atomic.AtomicReference

/**
 * Spark SessionExtensions hook for Quantum Engine. Recognizes a growing set
 * of LogicalPlan shapes and translates them to the engine's protobuf
 * PlanNode IR for native execution. Anything unrecognised falls through to
 * Spark's default planner.
 */
class QuantumExtension extends (SparkSessionExtensions => Unit) {
  override def apply(ext: SparkSessionExtensions): Unit = {
    ext.injectPlannerStrategy(session => new QuantumStrategy(session))
  }
}

class QuantumStrategy(session: org.apache.spark.sql.SparkSession = null)
    extends SparkStrategy {

  /** spark.quantum.agg.parallelism: > 1 hash-partitions GROUP BY input by
   *  the group keys and aggregates partitions in parallel on the engine's
   *  TaskScheduler (merge-free — groups are disjoint across partitions).
   *  0 (default) keeps the serial aggregate. */
  private def aggParallelism: Int = {
    if (session == null) 0
    else session.conf.get("spark.quantum.agg.parallelism", "0").toInt
  }

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = {
    QuantumStrategy.lastSeen.set(plan)
    QuantumStrategy.seenClasses.add(plan.getClass.getSimpleName)
    // The planner wraps the root plan in ReturnAnswer before applying
    // strategies, and Spark's SpecialLimits claims any ReturnAnswer it
    // sees (emitting CollectLimitExec / TakeOrderedAndProjectExec for
    // outer LIMIT / ORDER BY). Injected strategies run first, so matching
    // through ReturnAnswer here is what lets native LIMIT / SORT win.
    val (unwrapped, isRoot) = plan match {
      case ReturnAnswer(child) => (child, true)
      case other               => (other, false)
    }
    // Strip outer LIMIT wrappers and translate the inner plan with the
    // limit threaded through. Catalyst typically plans `LIMIT N` over an
    // Aggregate as `GlobalLimit(N, LocalLimit(N, Aggregate(...)))`, both
    // operating on the same logical limit; we collapse to one LimitNode.
    val (postLimit, limit) = QuantumStrategy.stripOuterLimits(unwrapped)
    // ORDER BY over a column that isn't in the SELECT list plans as
    // Project(selected, Sort(…, Aggregate(… + hidden ordering agg))) —
    // Catalyst adds the ordering aggregate to the Aggregate and projects
    // it back out above the Sort. Only attribute-shaped project lists are
    // claimed; the projection becomes a physicalIndex remap in QuantumExec.
    val (postProject, projectList) = postLimit match {
      case Project(pl, s: Sort) => (s: LogicalPlan, Some(pl))
      case other                => (other, None)
    }
    // ORDER BY plans as Sort above the Aggregate (below any LIMIT).
    // QuantumExec yields a single partition, so a partition-local sort is
    // also a total order — global=false is fine to claim.
    val (inner, sortOrder) = postProject match {
      case Sort(order, _, child) => (child, Some(order))
      case other                 => (other, None)
    }
    inner match {
      case Filter(having, agg: Aggregate) =>
        QuantumStrategy.tryTranslate(
            Some(agg), agg.child, postLimit.output, Some(having), limit, sortOrder,
            projectList, aggParallelism)
          .map(qe => Seq(qe))
          .getOrElse(Nil)
      case agg: Aggregate =>
        QuantumStrategy.tryTranslate(
            Some(agg), agg.child, postLimit.output, None, limit, sortOrder,
            projectList, aggParallelism)
          .map(quantumExec => Seq(quantumExec))
          .getOrElse(Nil)
      case other if isRoot =>
        // Plain (non-aggregate) query root: SELECT … WHERE … ORDER BY …
        // LIMIT without GROUP BY. Gated to the ReturnAnswer root so we
        // claim whole end-user queries, not arbitrary subtrees of plans
        // Spark is in the middle of planning.
        QuantumStrategy.tryTranslate(
            None, other, postLimit.output, None, limit, sortOrder, projectList)
          .map(quantumExec => Seq(quantumExec))
          .getOrElse(Nil)
      case _ => Nil
    }
  }
}

object QuantumStrategy {
  // Thread-local sinks for tests.
  val lastSeen: AtomicReference[LogicalPlan]  = new AtomicReference[LogicalPlan]()
  val lastTranslated: AtomicReference[String] = new AtomicReference[String]()
  val seenClasses: java.util.Set[String] =
    java.util.Collections.synchronizedSet(new java.util.LinkedHashSet[String]())
  // Append-only log of every translation result this strategy produced
  // during a query. Spark's planner may call `apply` multiple times on
  // different subtrees, so `lastTranslated` can be overwritten —
  // `translationLog` preserves every label.
  val translationLog: java.util.List[String] =
    java.util.Collections.synchronizedList(new java.util.ArrayList[String]())

  def reset(): Unit = {
    lastSeen.set(null)
    lastTranslated.set(null)
    seenClasses.clear()
    translationLog.clear()
  }
  def lastObservedPlan: Option[LogicalPlan] = Option(lastSeen.get())
  def lastTranslation: Option[String]       = Option(lastTranslated.get())
  def translations: Seq[String] = {
    import scala.jdk.CollectionConverters._
    translationLog.asScala.toSeq
  }
  def observedClasses: Set[String] = {
    import scala.jdk.CollectionConverters._
    seenClasses.asScala.toSet
  }

  /**
   * Result of recursively peeling a LogicalPlan subtree. `plan` is the
   * fully-built proto plan tree representing this subtree (scans / filters
   * / joins, already wired together). `idMap` maps every output `ExprId`
   * of the subtree to its column position in the plan's output. `output`
   * is the subtree's positional output schema. `sources` collects in-memory
   * LocalRelation data that must be registered with the engine before
   * `plan` can execute (each LocalRelation gets a fresh `sourceId`).
   *
   * Carrying `filterCount` is a diagnostic: the strategy reports how many
   * FilterNodes it pushed into the plan so tests can verify pushdown.
   */
  case class PeelResult(
      plan:        QuantumPlan.PlanNode,
      idMap:       Map[ExprId, Int],
      output:      Seq[Attribute],
      sources:     Seq[(String, () => Array[InputColumn])],
      filterCount: Int)

  /**
   * Strip outer GlobalLimit / LocalLimit wrappers, returning the inner
   * plan plus the tightest limit seen. Catalyst plans `LIMIT N` over an
   * Aggregate as `GlobalLimit(N, LocalLimit(N, Aggregate(...)))`; we
   * collapse both into one limit value to pass downstream.
   */
  def stripOuterLimits(plan: LogicalPlan): (LogicalPlan, Option[Long]) = {
    def extractLimit(e: Expression): Option[Long] = e match {
      case Literal(n: java.lang.Integer, IntegerType) => Some(n.longValue)
      case Literal(n: java.lang.Long, LongType)       => Some(n)
      case _                                          => None
    }
    plan match {
      case GlobalLimit(le, child) =>
        val (inner, innerLimit) = stripOuterLimits(child)
        val outer = extractLimit(le)
        (inner, (outer, innerLimit) match {
          case (Some(a), Some(b)) => Some(math.min(a, b))
          case (Some(a), None)    => Some(a)
          case (None, Some(b))    => Some(b)
          case _                  => None
        })
      case LocalLimit(le, child) =>
        val (inner, innerLimit) = stripOuterLimits(child)
        val outer = extractLimit(le)
        (inner, (outer, innerLimit) match {
          case (Some(a), Some(b)) => Some(math.min(a, b))
          case (Some(a), None)    => Some(a)
          case (None, Some(b))    => Some(b)
          case _                  => None
        })
      case other => (other, None)
    }
  }

  /**
   * Translate a recognised plan shape to a QuantumExec. With `aggOpt` set,
   * `peelTarget` is the Aggregate's child and a HashAggregateNode (plus
   * HAVING filters) is built above it. With `aggOpt` empty — a plain
   * query root — the peeled plan is used directly and `plainOutput` (the
   * root plan's output attributes) drives the output mapping.
   */
  def tryTranslate(
      aggOpt: Option[Aggregate],
      peelTarget: LogicalPlan,
      plainOutput: Seq[Attribute],
      havingCondition: Option[Expression] = None,
      limit: Option[Long] = None,
      sortOrder: Option[Seq[SortOrder]] = None,
      projectList: Option[Seq[NamedExpression]] = None,
      aggParallelism: Int = 0): Option[SparkPlan] = {

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    def protoTypeFor(t: DataType): QuantumPlan.TypeId = t match {
      case LongType    => QuantumPlan.TypeId.TYPE_INT64
      case IntegerType => QuantumPlan.TypeId.TYPE_INT64  // widened
      case DoubleType  => QuantumPlan.TypeId.TYPE_FLOAT64
      case StringType  => QuantumPlan.TypeId.TYPE_STRING
      case other       => throw new IllegalStateException(s"unsupported col type $other")
    }

    def isSupportedColType(t: DataType): Boolean = t match {
      case LongType | StringType | IntegerType | DoubleType => true
      case _                                                => false
    }

    val sourceIdCounter = new java.util.concurrent.atomic.AtomicLong(0)
    def nextSourceId(): String =
      s"quantum_src_${System.nanoTime()}_${sourceIdCounter.incrementAndGet()}"

    def flattenAnd(e: Expression): Seq[Expression] = e match {
      case And(l, r) => flattenAnd(l) ++ flattenAnd(r)
      case other     => Seq(other)
    }

    def stripRedundantIsNotNull(conjuncts: Seq[Expression]): Seq[Expression] = {
      // A column referenced in any null-propagating predicate is implicitly
      // also IsNotNull — the predicate evaluates to NULL on a NULL input,
      // and Filter excludes NULL-predicate rows. Spark adds explicit
      // `isnotnull(col)` to enable parquet page-skipping; we strip it
      // when redundant.
      def referencedCols(e: Expression): Seq[ExprId] = e match {
        case GreaterThan(l, r)        => referencedCols(l) ++ referencedCols(r)
        case LessThan(l, r)           => referencedCols(l) ++ referencedCols(r)
        case EqualTo(l, r)            => referencedCols(l) ++ referencedCols(r)
        case GreaterThanOrEqual(l, r) => referencedCols(l) ++ referencedCols(r)
        case LessThanOrEqual(l, r)    => referencedCols(l) ++ referencedCols(r)
        case Like(c, _, _)            => referencedCols(c)
        case StartsWith(c, _)         => referencedCols(c)
        case EndsWith(c, _)           => referencedCols(c)
        case Contains(c, _)           => referencedCols(c)
        case ar: AttributeReference   => Seq(ar.exprId)
        case _                        => Seq.empty
      }
      val referenced: Set[ExprId] = conjuncts.flatMap(referencedCols).toSet
      conjuncts.filter {
        case IsNotNull(ar: AttributeReference) => !referenced.contains(ar.exprId)
        case _                                 => true
      }
    }

    def translateScalarWith(idMap: Map[ExprId, Int])(e: Expression): Option[QuantumPlan.ScalarExpr] = {
      def binOp(op: QuantumPlan.BinaryOpType, l: Expression, r: Expression) =
        for {
          ls <- translateScalarWith(idMap)(l)
          rs <- translateScalarWith(idMap)(r)
        } yield QuantumPlan.ScalarExpr.newBuilder()
          .setBinary(QuantumPlan.BinaryOp.newBuilder().setOp(op).setLhs(ls).setRhs(rs))
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
            .setLiteral(QuantumPlan.Literal.newBuilder().setInt64Value(v)).build())
        case Literal(v: java.lang.Integer, IntegerType) =>
          Some(QuantumPlan.ScalarExpr.newBuilder()
            .setLiteral(QuantumPlan.Literal.newBuilder().setInt64Value(v.longValue)).build())
        case Literal(v: UTF8String, StringType) =>
          Some(QuantumPlan.ScalarExpr.newBuilder()
            .setLiteral(QuantumPlan.Literal.newBuilder().setStringValue(v.toString)).build())
        case Literal(v: java.lang.Double, DoubleType) =>
          Some(QuantumPlan.ScalarExpr.newBuilder()
            .setLiteral(QuantumPlan.Literal.newBuilder().setDoubleValue(v)).build())
        case Literal(v: java.lang.Boolean, BooleanType) =>
          Some(QuantumPlan.ScalarExpr.newBuilder()
            .setLiteral(QuantumPlan.Literal.newBuilder().setBoolValue(v)).build())
        case GreaterThan(l, r)        => binOp(QuantumPlan.BinaryOpType.BINOP_GT,  l, r)
        case LessThan(l, r)           => binOp(QuantumPlan.BinaryOpType.BINOP_LT,  l, r)
        case EqualTo(l, r)            => binOp(QuantumPlan.BinaryOpType.BINOP_EQ,  l, r)
        case GreaterThanOrEqual(l, r) => binOp(QuantumPlan.BinaryOpType.BINOP_GTE, l, r)
        case LessThanOrEqual(l, r)    => binOp(QuantumPlan.BinaryOpType.BINOP_LTE, l, r)
        case And(l, r)                => binOp(QuantumPlan.BinaryOpType.BINOP_AND, l, r)
        case Or(l, r)                 => binOp(QuantumPlan.BinaryOpType.BINOP_OR,  l, r)
        case Not(c) =>
          translateScalarWith(idMap)(c).map { x =>
            QuantumPlan.ScalarExpr.newBuilder()
              .setUnary(QuantumPlan.UnaryOp.newBuilder()
                .setOp(QuantumPlan.UnaryOpType.UNOP_NOT).setInput(x)).build()
          }
        case IsNull(c) =>
          translateScalarWith(idMap)(c).map { x =>
            QuantumPlan.ScalarExpr.newBuilder()
              .setUnary(QuantumPlan.UnaryOp.newBuilder()
                .setOp(QuantumPlan.UnaryOpType.UNOP_IS_NULL).setInput(x)).build()
          }
        case IsNotNull(c) =>
          translateScalarWith(idMap)(c).map { x =>
            val isNull = QuantumPlan.ScalarExpr.newBuilder()
              .setUnary(QuantumPlan.UnaryOp.newBuilder()
                .setOp(QuantumPlan.UnaryOpType.UNOP_IS_NULL).setInput(x)).build()
            QuantumPlan.ScalarExpr.newBuilder()
              .setUnary(QuantumPlan.UnaryOp.newBuilder()
                .setOp(QuantumPlan.UnaryOpType.UNOP_NOT).setInput(isNull)).build()
          }
        case In(value, list) if list.nonEmpty =>
          // Rewrite `value IN (a, b, c)` as
          // `(value = a) OR (value = b) OR (value = c)`.
          // Catalyst would normally do this when the list is small enough,
          // but it doesn't always — e.g. for SQL `IN (literal_list)` over
          // an in-memory relation. Handling it here makes the strategy
          // robust to either shape.
          val rewritten: Expression = list.tail.foldLeft[Expression](EqualTo(value, list.head)) {
            (acc, x) => Or(acc, EqualTo(value, x))
          }
          translateScalarWith(idMap)(rewritten)
        case Like(child, Literal(pattern: UTF8String, StringType), _) =>
          // The engine's LikeOp takes a constant `std::string` pattern with
          // SQL `%` / `_` wildcards. We don't honor Catalyst's escape char
          // (Phase 3 engine doesn't support escapes); the pattern is
          // passed through verbatim.
          translateScalarWith(idMap)(child).map { c =>
            buildLike(c, pattern.toString)
          }
        case StartsWith(child, Literal(prefix: UTF8String, StringType)) =>
          // Catalyst rewrites `col LIKE 'foo%'` to StartsWith(col, 'foo')
          // when the pattern's wildcards are only a trailing `%`.
          translateScalarWith(idMap)(child).map { c =>
            buildLike(c, prefix.toString + "%")
          }
        case EndsWith(child, Literal(suffix: UTF8String, StringType)) =>
          // Catalyst rewrites `col LIKE '%foo'` to EndsWith(col, 'foo').
          translateScalarWith(idMap)(child).map { c =>
            buildLike(c, "%" + suffix.toString)
          }
        case Contains(child, Literal(needle: UTF8String, StringType)) =>
          // Catalyst rewrites `col LIKE '%foo%'` to Contains(col, 'foo').
          translateScalarWith(idMap)(child).map { c =>
            buildLike(c, "%" + needle.toString + "%")
          }
        case _ => None
      }
    }

    def buildLike(input: QuantumPlan.ScalarExpr, pattern: String): QuantumPlan.ScalarExpr =
      QuantumPlan.ScalarExpr.newBuilder()
        .setLike(QuantumPlan.LikeOp.newBuilder().setInput(input).setPattern(pattern))
        .build()

    // Marshal rows (LocalRelation data or a collected AQE stage) into
    // column-major typed arrays for the bridge.
    def marshalRows(output: Seq[Attribute], rows: Seq[InternalRow]): Array[InputColumn] = {
      val data  = rows.toArray
      val nCols = output.size
      Array.tabulate(nCols) { c =>
        def validityOrNull: Array[Byte] = {
          if (!data.exists(_.isNullAt(c))) null
          else data.map(r => if (r.isNullAt(c)) 0.toByte else 1.toByte)
        }
        output(c).dataType match {
          case LongType =>
            LongColumn(data.map(r => if (r.isNullAt(c)) 0L else r.getLong(c)),
                       validityOrNull)
          case IntegerType =>
            LongColumn(data.map(r => if (r.isNullAt(c)) 0L else r.getInt(c).toLong),
                       validityOrNull)
          case DoubleType =>
            DoubleColumn(data.map(r => if (r.isNullAt(c)) 0.0 else r.getDouble(c)),
                         validityOrNull)
          case StringType =>
            StringColumn(data.map(r => if (r.isNullAt(c)) null else r.getString(c)))
          case other =>
            throw new IllegalStateException(s"unsupported source col type $other")
        }
      }
    }

    // Build a FileScanNode (including partition columns) from a Parquet
    // HadoopFsRelation. Returns the proto node + the relation's output
    // attribute schema in column order.
    def buildFileScanNode(hr: HadoopFsRelation, output: Seq[Attribute]):
        Option[QuantumPlan.PlanNode] = {
      hr.fileFormat match {
        case _: ParquetFileFormat =>
          val partDirs        = hr.location.listFiles(Nil, Nil)
          val partitionFields = hr.partitionSchema.fields.toSeq
          val flat: Seq[(String, Seq[Any])] = partDirs.flatMap { pd =>
            val vals: Seq[Any] = partitionFields.indices.map { i =>
              if (pd.values.isNullAt(i)) null
              else pd.values.get(i, partitionFields(i).dataType)
            }
            pd.files.map(f => (f.getPath.toString, vals))
          }.sortBy(_._1)
          val paths        = flat.map(_._1)
          val partValuesPP = flat.map(_._2)
          if (paths.isEmpty) {
            lastTranslated.set("reject: Parquet LogicalRelation has no files")
            None
          } else {
            val partitionCols = output.takeRight(partitionFields.size)
            val fs = QuantumPlan.FileScanNode.newBuilder().setFormat("parquet")
            paths.foreach(fs.addPaths)
            if (partitionCols.nonEmpty) {
              val partSchemaB = QuantumPlan.Schema.newBuilder()
              partitionCols.foreach(a =>
                partSchemaB.addFields(QuantumPlan.Field.newBuilder()
                  .setName(a.name).setType(protoTypeFor(a.dataType))))
              fs.setPartitionColumns(partSchemaB)
              partValuesPP.foreach { row =>
                val pvRow = QuantumPlan.PartitionValueRow.newBuilder()
                row.zip(partitionCols).foreach { case (v, attr) =>
                  val litB = QuantumPlan.Literal.newBuilder()
                  (v, attr.dataType) match {
                    case (null, _) =>
                      attr.dataType match {
                        case LongType    => litB.setInt64Value(0L)
                        case IntegerType => litB.setInt64Value(0L)
                        case DoubleType  => litB.setDoubleValue(0.0)
                        case BooleanType => litB.setBoolValue(false)
                        case StringType  => litB.setStringValue("")
                        case _           =>
                      }
                    case (n: java.lang.Long,    LongType)    => litB.setInt64Value(n)
                    case (n: java.lang.Integer, IntegerType) => litB.setInt64Value(n.longValue)
                    case (n: java.lang.Double,  DoubleType)  => litB.setDoubleValue(n)
                    case (b: java.lang.Boolean, BooleanType) => litB.setBoolValue(b)
                    case (u: UTF8String,        StringType)  => litB.setStringValue(u.toString)
                    case (s: String,            StringType)  => litB.setStringValue(s)
                    case _ =>
                      throw new IllegalStateException(
                        s"Unsupported partition value $v for column ${attr.name}:${attr.dataType}")
                  }
                  pvRow.addValues(litB)
                }
                fs.addPartitionValues(pvRow)
              }
            }
            Some(QuantumPlan.PlanNode.newBuilder().setFileScan(fs).build())
          }
        case other =>
          lastTranslated.set(s"reject: unsupported HadoopFsRelation format $other")
          None
      }
    }

    // Translate Spark's JoinType to our proto enum. Naming-convention twist:
    // the native engine's `LEFT_OUTER` preserves the PROBE side (emits probe
    // rows that didn't match with NULL build cols), and `RIGHT_OUTER`
    // preserves the BUILD side. Our Spark strategy puts Spark's LEFT input
    // into `build` and Spark's RIGHT into `probe`, so to preserve Spark's
    // LEFT we want the engine to preserve build -> JOIN_RIGHT_OUTER; to
    // preserve Spark's RIGHT we want the engine to preserve probe ->
    // JOIN_LEFT_OUTER. Spark's INNER and FULL OUTER map straight through.
    // LeftSemi / LeftAnti / Cross decline.
    def translateJoinType(jt: JoinType): Option[QuantumPlan.JoinType] = jt match {
      case Inner      => Some(QuantumPlan.JoinType.JOIN_INNER)
      case LeftOuter  => Some(QuantumPlan.JoinType.JOIN_RIGHT_OUTER)
      case RightOuter => Some(QuantumPlan.JoinType.JOIN_LEFT_OUTER)
      case FullOuter  => Some(QuantumPlan.JoinType.JOIN_FULL_OUTER)
      case _          => None
    }

    // Extract equi-join keys from a Join condition. Each conjunct must be
    // `EqualTo(leftRef, rightRef)` (or its mirror). Returns a Seq of
    // (build-side col index, probe-side col index) using the *post-side-peel*
    // local indices on each side.
    def extractEquiKeys(
        condition: Expression,
        leftIdMap:  Map[ExprId, Int],
        rightIdMap: Map[ExprId, Int]): Option[Seq[(Int, Int)]] = {
      val conjuncts = flattenAnd(condition)
      val pairs: Seq[Option[(Int, Int)]] = conjuncts.map {
        case EqualTo(l: AttributeReference, r: AttributeReference) =>
          (leftIdMap.get(l.exprId), rightIdMap.get(r.exprId)) match {
            case (Some(li), Some(ri)) => Some((li, ri))
            case _ =>
              // Try mirrored sides.
              (leftIdMap.get(r.exprId), rightIdMap.get(l.exprId)) match {
                case (Some(li), Some(ri)) => Some((li, ri))
                case _                    => None
              }
          }
        case _ => None
      }
      if (pairs.exists(_.isEmpty)) None else Some(pairs.flatten)
    }

    // ------------------------------------------------------------------
    // Recursive peel. Builds the proto plan tree bottom-up.
    // ------------------------------------------------------------------

    def peel(p: LogicalPlan): Option[PeelResult] = p match {
      case lr: LocalRelation =>
        if (!lr.output.forall(a => isSupportedColType(a.dataType))) {
          lastTranslated.set(s"reject: LocalRelation has unsupported column types (" +
            lr.output.map(a => s"${a.name}:${a.dataType}").mkString(", ") + ")")
          None
        } else {
          val sid     = nextSourceId()
          val schemaB = QuantumPlan.Schema.newBuilder()
          lr.output.zipWithIndex.foreach { case (a, i) =>
            schemaB.addFields(QuantumPlan.Field.newBuilder()
              .setName(if (a.name.nonEmpty) a.name else s"c$i")
              .setType(protoTypeFor(a.dataType)))
          }
          val scan = QuantumPlan.PlanNode.newBuilder()
            .setScan(QuantumPlan.ScanNode.newBuilder().setSourceId(sid).setSchema(schemaB))
            .build()
          val idMap = lr.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
          Some(PeelResult(scan, idMap, lr.output,
                          Seq(sid -> (() => marshalRows(lr.output, lr.data))), 0))
        }

      case lqs: LogicalQueryStage =>
        // AQE hands re-optimized plans a materialized shuffle stage. When
        // the stage is a PURE reshuffle (an explicit repartition), its rows
        // are raw child rows — safe to treat as a leaf source, collected
        // from the already-finished stage at execution time. Stages that
        // contain partial aggregation must decline: re-aggregating partial
        // buffers (e.g. COUNT partials) under the original logical
        // Aggregate would double-count.
        lqs.logicalPlan match {
          case _: Repartition | _: RepartitionByExpression =>
            if (!lqs.output.forall(a => isSupportedColType(a.dataType))) {
              lastTranslated.set("reject: LogicalQueryStage has unsupported column types")
              None
            } else {
              val sid     = nextSourceId()
              val schemaB = QuantumPlan.Schema.newBuilder()
              lqs.output.zipWithIndex.foreach { case (a, i) =>
                schemaB.addFields(QuantumPlan.Field.newBuilder()
                  .setName(if (a.name.nonEmpty) a.name else s"c$i")
                  .setType(protoTypeFor(a.dataType)))
              }
              val scan = QuantumPlan.PlanNode.newBuilder()
                .setScan(QuantumPlan.ScanNode.newBuilder().setSourceId(sid).setSchema(schemaB))
                .build()
              val idMap = lqs.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
              val stagePlan = lqs.physicalPlan
              Some(PeelResult(scan, idMap, lqs.output,
                              Seq(sid -> (() => marshalRows(lqs.output,
                                  stagePlan.executeCollect().toSeq))), 0))
            }
          case other =>
            lastTranslated.set(
              s"reject: LogicalQueryStage over ${other.getClass.getSimpleName} " +
              "(only pure-reshuffle stages are safe to consume)")
            None
        }

      case LogicalRelation(hr: HadoopFsRelation, output, _, _) =>
        if (!output.forall(a => isSupportedColType(a.dataType))) {
          lastTranslated.set(s"reject: LogicalRelation has unsupported column types (" +
            output.map(a => s"${a.name}:${a.dataType}").mkString(", ") + ")")
          None
        } else {
          buildFileScanNode(hr, output).map { plan =>
            val idMap = output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
            PeelResult(plan, idMap, output, Seq.empty, 0)
          }
        }

      case sa: SubqueryAlias =>
        peel(sa.child)

      case pr: Project =>
        // We don't actually emit a ProjectNode in the proto plan — the
        // underlying scan still produces all of its original columns. Spark's
        // column-pruning Project is collapsed into an idMap rebind: kept
        // exprIds map to their original column positions; dropped exprIds
        // simply disappear from the map. peel.output stays as the scan's
        // natural output so join output_names matches the plan's actual
        // column count.
        peel(pr.child).flatMap { result =>
          val pairs: Seq[Option[(ExprId, Int)]] = pr.projectList.zip(pr.output).map {
            case (ar: AttributeReference, outAttr) =>
              result.idMap.get(ar.exprId).map(outAttr.exprId -> _)
            case (Alias(ar: AttributeReference, _), outAttr) =>
              result.idMap.get(ar.exprId).map(outAttr.exprId -> _)
            case _ => None
          }
          if (pairs.exists(_.isEmpty)) None
          else Some(result.copy(idMap = pairs.flatten.toMap))
        }

      case Union(children, _, _) =>
        // UNION ALL: recursively peel each child, combine sources, emit a
        // UnionAllNode wrapping the per-child plans. The Union's output
        // schema is the first child's; idMap uses the Union node's own
        // output attributes (Catalyst makes them positionally aligned
        // across children) so the parent operator can reference them.
        val peeled: Seq[Option[PeelResult]] = children.map(peel)
        if (peeled.exists(_.isEmpty)) {
          lastTranslated.set("reject: at least one Union child didn't peel")
          None
        } else {
          val rs = peeled.flatten
          val unionB = QuantumPlan.UnionAllNode.newBuilder()
          rs.foreach(r => unionB.addChildren(r.plan))
          val unionPlan = QuantumPlan.PlanNode.newBuilder().setUnionAll(unionB).build()
          // Union output: Catalyst's `Union.output` is positional across
          // children with the first child's exprIds (canonical). Use that
          // as our peel.output and remap idMap accordingly.
          val unionOutput = children.head.output  // same positional schema as Union.output
          val idMap = unionOutput.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
          Some(PeelResult(
            plan        = unionPlan,
            idMap       = idMap,
            output      = unionOutput,
            sources     = rs.flatMap(_.sources),
            filterCount = rs.map(_.filterCount).sum))
        }

      case Sort(order, _, child) =>
        // ORDER BY: peel the child then wrap in a SortNode. Each SortOrder
        // is (Expression, direction, nullOrdering). Inner expression is
        // typically an AttributeReference that we resolve via the child's
        // idMap to a column index; non-AttributeReference sort keys
        // (function calls, arithmetic) currently make us decline.
        peel(child).flatMap { result =>
          val translated: Seq[Option[QuantumPlan.ScalarExpr]] = order.map { so =>
            translateScalarWith(result.idMap)(so.child)
          }
          if (translated.exists(_.isEmpty)) {
            lastTranslated.set(s"reject: sort key untranslatable: $order")
            None
          } else {
            val sortB = QuantumPlan.SortNode.newBuilder().setChild(result.plan)
            translated.flatten.foreach(sortB.addKeys)
            order.foreach { so =>
              sortB.addAscending(so.direction == Ascending)
              sortB.addNullsFirst(so.nullOrdering match {
                case org.apache.spark.sql.catalyst.expressions.NullsFirst => true
                case _ => false
              })
            }
            val sortPlan = QuantumPlan.PlanNode.newBuilder().setSort(sortB).build()
            Some(result.copy(plan = sortPlan))
          }
        }

      case f: Filter =>
        peel(f.child).flatMap { result =>
          val conjs      = stripRedundantIsNotNull(flattenAnd(f.condition))
          val translated = conjs.map(translateScalarWith(result.idMap))
          if (translated.exists(_.isEmpty)) {
            lastTranslated.set(s"reject: predicate '${f.condition}' untranslatable")
            None
          } else {
            val wrapped = translated.flatten.foldLeft(result.plan) { (acc, pred) =>
              QuantumPlan.PlanNode.newBuilder()
                .setFilter(QuantumPlan.FilterNode.newBuilder().setChild(acc).setPredicate(pred))
                .build()
            }
            Some(result.copy(
              plan        = wrapped,
              filterCount = result.filterCount + translated.size))
          }
        }

      case j: Join =>
        // Equi-joins only. The Join's condition must be an AND-chain of
        // EqualTo(leftAttr, rightAttr) pairs.
        translateJoinType(j.joinType) match {
          case None =>
            lastTranslated.set(s"reject: unsupported join type ${j.joinType}")
            None
          case Some(protoJt) =>
            val cond = j.condition.getOrElse {
              lastTranslated.set("reject: join without condition (cross join unsupported)")
              return None
            }
            for {
              leftR  <- peel(j.left)
              rightR <- peel(j.right)
              keys   <- extractEquiKeys(cond, leftR.idMap, rightR.idMap).orElse {
                lastTranslated.set(s"reject: non-equi join condition: $cond")
                None
              }
            } yield {
              val joinB = QuantumPlan.HashJoinNode.newBuilder()
                .setBuild(leftR.plan)
                .setProbe(rightR.plan)
                .setJoinType(protoJt)
              keys.foreach { case (l, r) =>
                joinB.addBuildKeys(QuantumPlan.ScalarExpr.newBuilder()
                  .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(l)).build())
                joinB.addProbeKeys(QuantumPlan.ScalarExpr.newBuilder()
                  .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(r)).build())
              }
              // Output names — left cols followed by right cols.
              (leftR.output ++ rightR.output).foreach(a => joinB.addOutputNames(a.name))

              val joinPlan = QuantumPlan.PlanNode.newBuilder().setHashJoin(joinB).build()

              val combinedIdMap =
                leftR.idMap ++ rightR.idMap.view.mapValues(_ + leftR.output.size).toMap

              PeelResult(
                plan        = joinPlan,
                idMap       = combinedIdMap,
                output      = leftR.output ++ rightR.output,
                sources     = leftR.sources ++ rightR.sources,
                filterCount = leftR.filterCount + rightR.filterCount)
            }
        }

      case _ => None
    }

    // ------------------------------------------------------------------
    // tryTranslate body — peel the target, then build the HashAggregate
    // (or pass the peeled plan through for plain query roots).
    // ------------------------------------------------------------------

    val peeled: PeelResult = peel(peelTarget) match {
      case Some(r) => r
      case None    =>
        if (lastTranslated.get == null || !lastTranslated.get.startsWith("reject:"))
          lastTranslated.set("reject: plan didn't peel\n" + peelTarget.treeString)
        return None
    }
    val exprIdToCol = peeled.idMap

    // Build the plan above the peeled child and the column map that's
    // valid above it: `outMap` resolves exprIds to engine output columns
    // for sort keys and the final output mapping; `defaultItems` is the
    // output when no outer Project was stripped.
    val (root0, outMap, defaultItems, labelPrefix):
        (QuantumPlan.PlanNode, Map[ExprId, Int], Seq[NamedExpression], String) =
      aggOpt match {
      case None =>
        (peeled.plan, peeled.idMap, plainOutput,
         s"Plain(${plainOutput.size} outputs, ${peeled.filterCount} pushed filters, ")
      case Some(agg) =>
        val groupingExprs  = agg.groupingExpressions
        val aggregateExprs = agg.aggregateExpressions

    // Grouping
    val groupColIndices: Seq[Int] = groupingExprs.flatMap {
      case ar: AttributeReference => exprIdToCol.get(ar.exprId)
      case _                      => None
    }
    if (groupColIndices.size != groupingExprs.size) {
      lastTranslated.set("reject: grouping exprs unsupported")
      return None
    }
    val groupKeySet: Set[ExprId] = groupingExprs.collect {
      case ar: AttributeReference => ar.exprId
    }.toSet

    // Output items: group keys or aggregates.
    sealed trait OutputItem
    case class GroupKeyOut(colIdx: Int) extends OutputItem
    case class AggOut(spec: QuantumPlan.AggregateSpec, label: String) extends OutputItem

    def aggCol(idx: Int): QuantumPlan.ScalarExpr =
      QuantumPlan.ScalarExpr.newBuilder()
        .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(idx)).build()

    def specOf(kind: QuantumPlan.AggregateKind, inputCol: Option[Int],
               inputType: QuantumPlan.TypeId = QuantumPlan.TypeId.TYPE_INT64):
        QuantumPlan.AggregateSpec = {
      val b = QuantumPlan.AggregateSpec.newBuilder()
        .setKind(kind).setInputType(inputType)
      inputCol.foreach(idx => b.setInput(aggCol(idx)))
      b.build()
    }

    def aggTypeFor(t: DataType): Option[QuantumPlan.TypeId] = t match {
      case LongType    => Some(QuantumPlan.TypeId.TYPE_INT64)
      case DoubleType  => Some(QuantumPlan.TypeId.TYPE_FLOAT64)
      case _           => None
    }

    def classifyAgg(fn: AggregateFunction): Option[OutputItem] = fn match {
      case Sum(ar: AttributeReference, _) =>
        for { it  <- aggTypeFor(ar.dataType)
              idx <- exprIdToCol.get(ar.exprId) }
          yield AggOut(specOf(QuantumPlan.AggregateKind.AGG_SUM, Some(idx), it), "sum")
      case Min(ar: AttributeReference) =>
        for { it  <- aggTypeFor(ar.dataType)
              idx <- exprIdToCol.get(ar.exprId) }
          yield AggOut(specOf(QuantumPlan.AggregateKind.AGG_MIN, Some(idx), it), "min")
      case Max(ar: AttributeReference) =>
        for { it  <- aggTypeFor(ar.dataType)
              idx <- exprIdToCol.get(ar.exprId) }
          yield AggOut(specOf(QuantumPlan.AggregateKind.AGG_MAX, Some(idx), it), "max")
      case Average(ar: AttributeReference, _) =>
        for { it  <- aggTypeFor(ar.dataType)
              idx <- exprIdToCol.get(ar.exprId) }
          yield AggOut(specOf(QuantumPlan.AggregateKind.AGG_AVG, Some(idx), it), "avg")
      case Count(children) if children.forall { case _: Literal => true; case _ => false } =>
        Some(AggOut(specOf(QuantumPlan.AggregateKind.AGG_COUNT_STAR, None), "count_star"))
      case Count(Seq(ar: AttributeReference)) =>
        exprIdToCol.get(ar.exprId).map(i =>
          AggOut(specOf(QuantumPlan.AggregateKind.AGG_COUNT, Some(i)), "count"))
      case _ => None
    }

    // Map an input column's DataType to the proto TypeId for distinct
    // counting (INT64 / STRING / FLOAT64 supported).
    def distinctInputType(t: DataType): Option[QuantumPlan.TypeId] = t match {
      case LongType    => Some(QuantumPlan.TypeId.TYPE_INT64)
      case StringType  => Some(QuantumPlan.TypeId.TYPE_STRING)
      case DoubleType  => Some(QuantumPlan.TypeId.TYPE_FLOAT64)
      case _           => None
    }

    // COUNT(DISTINCT col) — the isDistinct flag is on AggregateExpression,
    // not on the wrapped Count function, so classifyAgg can't see it. Caller
    // peels the AggregateExpression and routes isDistinct through here.
    def classifyDistinctAgg(fn: AggregateFunction): Option[OutputItem] = fn match {
      case Count(Seq(ar: AttributeReference)) =>
        for {
          it  <- distinctInputType(ar.dataType)
          idx <- exprIdToCol.get(ar.exprId)
        } yield AggOut(
          specOf(QuantumPlan.AggregateKind.AGG_COUNT_DISTINCT, Some(idx), it),
          "count_distinct")
      case _ => None
    }

    val outputItems: Seq[OutputItem] = aggregateExprs.flatMap {
      case ar: AttributeReference if groupKeySet.contains(ar.exprId) =>
        exprIdToCol.get(ar.exprId).map(GroupKeyOut)
      case Alias(ar: AttributeReference, _) if groupKeySet.contains(ar.exprId) =>
        exprIdToCol.get(ar.exprId).map(GroupKeyOut)
      case Alias(AggregateExpression(fn, _, true,  _, _), _) => classifyDistinctAgg(fn)
      case Alias(AggregateExpression(fn, _, false, _, _), _) => classifyAgg(fn)
      case other =>
        lastTranslated.set(s"reject: unsupported output expression: $other")
        return None
    }
    if (outputItems.size != aggregateExprs.size) {
      lastTranslated.set("reject: some output expressions couldn't be classified")
      return None
    }

    val hashAggB = QuantumPlan.HashAggregateNode.newBuilder().setChild(peeled.plan)
    if (aggParallelism > 1) hashAggB.setParallelism(aggParallelism)
    groupColIndices.foreach(idx => hashAggB.addGroupKeys(aggCol(idx)))
    val orderedAggs = outputItems.collect { case AggOut(spec, _) => spec }
    orderedAggs.foreach(hashAggB.addAggregates)
    val groupNames = groupingExprs.indices.map(i => s"g$i")
    val aggNames   = outputItems.collect { case AggOut(_, label) => label }
    (groupNames ++ aggNames).foreach(hashAggB.addOutputNames)

    val aggPlan = QuantumPlan.PlanNode.newBuilder().setHashAgg(hashAggB).build()

    // Physical-to-Spark output reordering.
    val physicalIndexBuilder = scala.collection.mutable.ArrayBuffer.empty[Int]
    val nGroupKeys = groupingExprs.size
    var aggSlot = 0
    outputItems.foreach {
      case GroupKeyOut(localColIdx) =>
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

    // HAVING pushdown. The Filter's condition references the Aggregate's
    // output Attributes by exprId; map each agg.output(i).exprId to its
    // engine column position via physicalIndex(i).
    val (root, havingPushed, havingDeclined) = havingCondition match {
      case None => (aggPlan, 0, false)
      case Some(cond) =>
        val havingIdMap: Map[ExprId, Int] =
          agg.output.zip(physicalIndex).map { case (attr, idx) => attr.exprId -> idx }.toMap
        val conjs      = stripRedundantIsNotNull(flattenAnd(cond))
        val translated = conjs.map(translateScalarWith(havingIdMap))
        if (translated.exists(_.isEmpty)) {
          // Decline the whole plan so Spark handles Filter(Aggregate)
          // itself via FilterExec(QuantumExec) — otherwise the HAVING
          // would be silently dropped.
          (aggPlan, 0, true)
        } else {
          val wrapped = translated.flatten.foldLeft(aggPlan) { (acc, pred) =>
            QuantumPlan.PlanNode.newBuilder()
              .setFilter(QuantumPlan.FilterNode.newBuilder().setChild(acc).setPredicate(pred))
              .build()
          }
          (wrapped, translated.size, false)
        }
    }

    if (havingDeclined) {
      lastTranslated.set("reject: HAVING untranslatable; Spark will handle filter natively")
      return None
    }

        val aggOutMap: Map[ExprId, Int] =
          agg.output.zip(physicalIndex).map { case (attr, idx) => attr.exprId -> idx }.toMap
        (root, aggOutMap, agg.output,
         "HashAggregate(" + aggregateExprs.size + " outputs over " +
           groupColIndices.size + " group keys, " + peeled.filterCount +
           " pushed filters, " + havingPushed + " HAVING filters, ")
    }

    // ORDER BY above root0: SortNode sits above the HAVING filters and
    // below the LimitNode (SQL semantics: HAVING → ORDER BY → LIMIT).
    // Declining here is safe — the planner re-applies strategies to the
    // subtrees via planLater, so an inner Aggregate still routes natively
    // under Spark's own SortExec.
    val (rootWithSort, sortPushed) = sortOrder match {
      case None        => (root0, false)
      case Some(order) =>
        val translated = order.map(so => translateScalarWith(outMap)(so.child))
        if (translated.exists(_.isEmpty)) {
          lastTranslated.set(s"reject: sort key untranslatable: $order")
          return None
        }
        val sortB = QuantumPlan.SortNode.newBuilder().setChild(root0)
        translated.flatten.foreach(sortB.addKeys)
        order.foreach { so =>
          sortB.addAscending(so.direction == Ascending)
          sortB.addNullsFirst(so.nullOrdering match {
            case org.apache.spark.sql.catalyst.expressions.NullsFirst => true
            case _ => false
          })
        }
        (QuantumPlan.PlanNode.newBuilder().setSort(sortB).build(), true)
    }

    // Wrap in a LimitNode if a LIMIT was stripped off the outer plan.
    val rootWithLimit: QuantumPlan.PlanNode = limit match {
      case None    => rootWithSort
      case Some(n) =>
        QuantumPlan.PlanNode.newBuilder()
          .setLimit(QuantumPlan.LimitNode.newBuilder().setChild(rootWithSort).setLimit(n))
          .build()
    }

    // Output mapping: each output item resolves to an engine column via
    // outMap. With a stripped outer Project this also drops/renames
    // columns (e.g. Catalyst's hidden ordering aggregates) — the engine
    // still emits every column; dropped ones are simply never read at the
    // JNA boundary. Only attribute-shaped items are supported; anything
    // else declines the whole match.
    val items: Seq[NamedExpression] = projectList.getOrElse(defaultItems)
    val mapped: Seq[Option[(Attribute, Int)]] = items.map {
      case a @ Alias(ar: AttributeReference, _) =>
        outMap.get(ar.exprId).map(i => (a.toAttribute, i))
      case ar: Attribute =>
        outMap.get(ar.exprId).map(i => (ar, i))
      case _ => None
    }
    if (mapped.exists(_.isEmpty)) {
      lastTranslated.set(s"reject: output expressions unmappable: $items")
      return None
    }
    val finalOutput        = mapped.flatten.map(_._1)
    val finalPhysicalIndex = mapped.flatten.map(_._2).toArray

    val label = labelPrefix +
      (if (sortPushed) s"ORDER BY ${sortOrder.get.size} keys, " else "") +
      limit.map(n => s"LIMIT $n, ").getOrElse("") +
      projectList.map(pl => s"project ${pl.size}/${defaultItems.size} cols, ").getOrElse("") +
      peeled.sources.size + " sources)"
    lastTranslated.set(label)
    translationLog.add(label)
    Some(QuantumExec(
      output        = finalOutput,
      sources       = peeled.sources,
      planBytes     = rootWithLimit.toByteArray,
      physicalIndex = finalPhysicalIndex))
  }
}
