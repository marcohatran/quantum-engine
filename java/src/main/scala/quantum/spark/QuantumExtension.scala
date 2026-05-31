package quantum.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.SparkStrategy

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
 * Phase 4.6 scope: the extension registers a planner strategy that observes
 * every LogicalPlan it sees but always returns `Nil` (so Spark falls back to
 * its default planner). The recorded plans drive the integration test. Full
 * plan rewriting — detecting eligible HashAggregate / HashJoin shapes and
 * handing them to the C ABI via the `quantum.QuantumEngine` JNA bindings —
 * is the next slice.
 */
class QuantumExtension extends (SparkSessionExtensions => Unit) {
  override def apply(ext: SparkSessionExtensions): Unit = {
    ext.injectPlannerStrategy(_ => new QuantumStrategy())
  }
}

/**
 * Planner strategy that observes every logical plan considered by the
 * physical planner. Records each plan into a thread-local sink so tests can
 * verify the extension actually fires.
 *
 * Returning `Nil` means "I have no SparkPlan candidate for this LogicalPlan"
 * which is the correct way to opt out — Spark continues with the next
 * strategy in the chain. The native-execution rewrite is the next slice.
 */
class QuantumStrategy extends SparkStrategy {
  override def apply(plan: LogicalPlan): Seq[SparkPlan] = {
    QuantumStrategy.lastSeen.set(plan)
    Nil
  }
}

object QuantumStrategy {
  // Thread-local sink for the most-recent logical plan the strategy saw —
  // tests inspect it to confirm the extension is wired into the session.
  val lastSeen: AtomicReference[LogicalPlan] = new AtomicReference[LogicalPlan]()

  def reset(): Unit = lastSeen.set(null)
  def lastObservedPlan: Option[LogicalPlan] = Option(lastSeen.get())
}
