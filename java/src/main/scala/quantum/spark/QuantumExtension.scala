package quantum.spark

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeReference, EqualTo, ExprId, Expression, GreaterThan, GreaterThanOrEqual, In, IsNotNull, IsNull, LessThan, LessThanOrEqual, Literal, NamedExpression, Not, Or}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, AggregateFunction, Average, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.plans.{FullOuter, Inner, JoinType, LeftOuter, RightOuter}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, LocalRelation, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
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
    ext.injectPlannerStrategy(_ => new QuantumStrategy())
  }
}

class QuantumStrategy extends SparkStrategy {

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = {
    QuantumStrategy.lastSeen.set(plan)
    QuantumStrategy.seenClasses.add(plan.getClass.getSimpleName)
    plan match {
      // HAVING: Catalyst plans `... GROUP BY k HAVING <cond>` as a Filter
      // wrapping the Aggregate. Translate the HAVING into a FilterNode
      // pushed above the HashAggregateNode. If untranslatable, decline so
      // Spark falls back to its own Filter on top of QuantumExec.
      case Filter(having, agg @ Aggregate(groupingExprs, aggregateExprs, child)) =>
        QuantumStrategy.tryTranslateLongAggregate(
            agg, groupingExprs, aggregateExprs, child, Some(having))
          .map(qe => Seq(qe))
          .getOrElse(Nil)
      case agg @ Aggregate(groupingExprs, aggregateExprs, child) =>
        QuantumStrategy.tryTranslateLongAggregate(agg, groupingExprs, aggregateExprs, child)
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

  def reset(): Unit = {
    lastSeen.set(null)
    lastTranslated.set(null)
    seenClasses.clear()
  }
  def lastObservedPlan: Option[LogicalPlan] = Option(lastSeen.get())
  def lastTranslation: Option[String]       = Option(lastTranslated.get())
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
      sources:     Seq[(String, Array[InputColumn])],
      filterCount: Int)

  def tryTranslateLongAggregate(
      agg: Aggregate,
      groupingExprs: Seq[Expression],
      aggregateExprs: Seq[NamedExpression],
      child: LogicalPlan,
      havingCondition: Option[Expression] = None): Option[SparkPlan] = {

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    def protoTypeFor(t: DataType): QuantumPlan.TypeId = t match {
      case LongType    => QuantumPlan.TypeId.TYPE_INT64
      case IntegerType => QuantumPlan.TypeId.TYPE_INT64  // widened
      case StringType  => QuantumPlan.TypeId.TYPE_STRING
      case other       => throw new IllegalStateException(s"unsupported col type $other")
    }

    def isSupportedColType(t: DataType): Boolean = t match {
      case LongType | StringType | IntegerType => true
      case _                                   => false
    }

    val sourceIdCounter = new java.util.concurrent.atomic.AtomicLong(0)
    def nextSourceId(): String =
      s"quantum_src_${System.nanoTime()}_${sourceIdCounter.incrementAndGet()}"

    def flattenAnd(e: Expression): Seq[Expression] = e match {
      case And(l, r) => flattenAnd(l) ++ flattenAnd(r)
      case other     => Seq(other)
    }

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
        case _ => None
      }
    }

    // Marshal a LocalRelation into column-major typed arrays for the JNA bridge.
    def marshalLocalRelation(lr: LocalRelation): Array[InputColumn] = {
      val data  = lr.data.toArray
      val nCols = lr.output.size
      Array.tabulate(nCols) { c =>
        def validityOrNull: Array[Byte] = {
          if (!data.exists(_.isNullAt(c))) null
          else data.map(r => if (r.isNullAt(c)) 0.toByte else 1.toByte)
        }
        lr.output(c).dataType match {
          case LongType =>
            LongColumn(data.map(r => if (r.isNullAt(c)) 0L else r.getLong(c)),
                       validityOrNull)
          case IntegerType =>
            LongColumn(data.map(r => if (r.isNullAt(c)) 0L else r.getInt(c).toLong),
                       validityOrNull)
          case StringType =>
            StringColumn(data.map(r => if (r.isNullAt(c)) null else r.getString(c)))
          case other =>
            throw new IllegalStateException(s"unsupported LR col type $other")
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
          val cols  = marshalLocalRelation(lr)
          Some(PeelResult(scan, idMap, lr.output, Seq(sid -> cols), 0))
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
    // tryTranslateLongAggregate body — peel the child then wrap in HashAgg.
    // ------------------------------------------------------------------

    val peeled: PeelResult = peel(child) match {
      case Some(r) => r
      case None    =>
        if (lastTranslated.get == null || !lastTranslated.get.startsWith("reject:"))
          lastTranslated.set("reject: child didn't peel\n" + child.treeString)
        return None
    }
    val exprIdToCol = peeled.idMap

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

    def specOf(kind: QuantumPlan.AggregateKind, inputCol: Option[Int]): QuantumPlan.AggregateSpec = {
      val b = QuantumPlan.AggregateSpec.newBuilder()
        .setKind(kind).setInputType(QuantumPlan.TypeId.TYPE_INT64)
      inputCol.foreach(idx => b.setInput(aggCol(idx)))
      b.build()
    }

    def classifyAgg(fn: AggregateFunction): Option[OutputItem] = fn match {
      case Sum(ar: AttributeReference, _) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(i => AggOut(specOf(QuantumPlan.AggregateKind.AGG_SUM, Some(i)), "sum"))
      case Min(ar: AttributeReference) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(i => AggOut(specOf(QuantumPlan.AggregateKind.AGG_MIN, Some(i)), "min"))
      case Max(ar: AttributeReference) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(i => AggOut(specOf(QuantumPlan.AggregateKind.AGG_MAX, Some(i)), "max"))
      case Average(ar: AttributeReference, _) if ar.dataType == LongType =>
        exprIdToCol.get(ar.exprId).map(i => AggOut(specOf(QuantumPlan.AggregateKind.AGG_AVG, Some(i)), "avg"))
      case Count(children) if children.forall { case _: Literal => true; case _ => false } =>
        Some(AggOut(specOf(QuantumPlan.AggregateKind.AGG_COUNT_STAR, None), "count_star"))
      case Count(Seq(ar: AttributeReference)) =>
        exprIdToCol.get(ar.exprId).map(i =>
          AggOut(specOf(QuantumPlan.AggregateKind.AGG_COUNT, Some(i)), "count"))
      case _ => None
    }

    val outputItems: Seq[OutputItem] = aggregateExprs.flatMap {
      case ar: AttributeReference if groupKeySet.contains(ar.exprId) =>
        exprIdToCol.get(ar.exprId).map(GroupKeyOut)
      case Alias(ar: AttributeReference, _) if groupKeySet.contains(ar.exprId) =>
        exprIdToCol.get(ar.exprId).map(GroupKeyOut)
      case Alias(AggregateExpression(fn, _, _, _, _), _) => classifyAgg(fn)
      case other =>
        lastTranslated.set(s"reject: unsupported output expression: $other")
        return None
    }
    if (outputItems.size != aggregateExprs.size) {
      lastTranslated.set("reject: some output expressions couldn't be classified")
      return None
    }

    val hashAggB = QuantumPlan.HashAggregateNode.newBuilder().setChild(peeled.plan)
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

    lastTranslated.set("HashAggregate(" + aggregateExprs.size + " outputs over " +
      groupColIndices.size + " group keys, " + peeled.filterCount + " pushed filters, " +
      havingPushed + " HAVING filters, " + peeled.sources.size + " sources)")
    Some(QuantumExec(
      output        = agg.output,
      sources       = peeled.sources,
      planBytes     = root.toByteArray,
      physicalIndex = physicalIndex))
  }
}
