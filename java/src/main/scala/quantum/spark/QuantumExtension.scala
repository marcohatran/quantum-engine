package quantum.spark

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Sum}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, LocalRelation, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.types.LongType
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
   * Phase 4.7 scope: recognise `Aggregate([k:Long], [Alias(k), Alias(Sum(v))],
   * LocalRelation(long k, long v))` exactly, and translate it to a Quantum
   * `HashAggregateNode(Scan, group=[col 0], aggs=[Sum(col 1)])` over the
   * LocalRelation's row data. Everything else falls through.
   */
  def tryTranslateLongAggregate(
      agg: Aggregate,
      groupingExprs: Seq[Expression],
      aggregateExprs: Seq[NamedExpression],
      child: LogicalPlan): Option[SparkPlan] = {

    // Peel off SubqueryAlias and column-renaming Projects to reach a
    // LocalRelation. `createOrReplaceTempView` + `spark.sql` usually wraps
    // the source in `SubqueryAlias(Project(LocalRelation))` for column
    // renaming; the data is still positionally identical.
    @scala.annotation.tailrec
    def peelToLocalRelation(p: LogicalPlan): Option[LocalRelation] = p match {
      case lr: LocalRelation => Some(lr)
      case sa: SubqueryAlias => peelToLocalRelation(sa.child)
      case pr: Project if pr.projectList.forall {
        case Alias(_: AttributeReference, _) => true
        case _: AttributeReference => true
        case _ => false
      } => peelToLocalRelation(pr.child)
      case _ => None
    }

    val localRel = peelToLocalRelation(child) match {
      case Some(lr) if lr.output.size == 2 &&
                       lr.output.forall(_.dataType == LongType) => lr
      case Some(lr) =>
        lastTranslated.set(s"reject: peeled to LocalRelation but schema mismatch " +
                           s"(${lr.output.size} cols, types=${lr.output.map(_.dataType).mkString(",")})")
        return None
      case None =>
        lastTranslated.set("reject: child didn't peel to LocalRelation\n" + child.treeString)
        return None
    }

    // Position-based pattern: grouping is a single Long attribute (col 0),
    // aggregate is the grouping key + SUM(other Long attribute) (col 1).
    val isLongAttr: PartialFunction[Expression, Boolean] = {
      case ar: AttributeReference => ar.dataType == LongType
    }
    if (!(groupingExprs.size == 1 && isLongAttr.applyOrElse[Expression, Boolean](groupingExprs.head, _ => false))) {
      lastTranslated.set(s"reject: grouping shape = ${groupingExprs.map(_.getClass.getSimpleName).mkString(",")}")
      return None
    }
    val outputAttrs = aggregateExprs match {
      case Seq(
        Alias(_: AttributeReference, _),
        Alias(AggregateExpression(Sum(_: AttributeReference, _), _, _, _, _), _)) =>
        agg.output
      case _ =>
        lastTranslated.set(s"reject: agg shape =\n" +
          aggregateExprs.zipWithIndex.map { case (e, i) =>
            s"  [$i] ${e.getClass.getSimpleName}: $e"
          }.mkString("\n"))
        return None
    }

    // Extract column-major Long data. Column ordering is preserved through
    // the peeled-off Project chain since we only allow trivial AliasOf or
    // bare AttributeReference projection items.
    val keyCol: Array[Long] = localRel.data.iterator.map(_.getLong(0)).toArray
    val valCol: Array[Long] = localRel.data.iterator.map(_.getLong(1)).toArray

    val schema = QuantumPlan.Schema.newBuilder()
      .addFields(QuantumPlan.Field.newBuilder().setName("k").setType(QuantumPlan.TypeId.TYPE_INT64))
      .addFields(QuantumPlan.Field.newBuilder().setName("v").setType(QuantumPlan.TypeId.TYPE_INT64))
      .build()
    val sourceId = s"quantum_src_${System.nanoTime()}"
    val scan = QuantumPlan.PlanNode.newBuilder()
      .setScan(QuantumPlan.ScanNode.newBuilder().setSourceId(sourceId).setSchema(schema))
      .build()
    val colK = QuantumPlan.ScalarExpr.newBuilder()
      .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(0)).build()
    val colV = QuantumPlan.ScalarExpr.newBuilder()
      .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(1)).build()
    val sumSpec = QuantumPlan.AggregateSpec.newBuilder()
      .setKind(QuantumPlan.AggregateKind.AGG_SUM)
      .setInputType(QuantumPlan.TypeId.TYPE_INT64)
      .setInput(colV).build()
    val root = QuantumPlan.PlanNode.newBuilder()
      .setHashAgg(QuantumPlan.HashAggregateNode.newBuilder()
        .setChild(scan)
        .addGroupKeys(colK)
        .addAggregates(sumSpec)
        .addOutputNames("k").addOutputNames("sum_v"))
      .build()

    lastTranslated.set("HashAggregate(sum_long_by_long)")
    Some(QuantumExec(
      output       = outputAttrs,
      inputColumns = Array(keyCol, valCol),
      sourceId     = sourceId,
      planBytes    = root.toByteArray))
  }
}
