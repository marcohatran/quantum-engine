package quantum;

import com.google.protobuf.ByteString;
import org.junit.jupiter.api.Test;
import quantum.plan.proto.QuantumPlan;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Drives the native engine through the JNA bridge: builds a PlanNode in
 * Java via the generated protobuf classes, registers a source, executes,
 * iterates result batches, verifies values.
 */
class QuantumEngineEndToEndTest {

    /**
     * SELECT k, SUM(v), COUNT(*) FROM kv GROUP BY k
     * over a single batch of 5 (k,v) int64 rows.
     */
    @Test
    void groupBySumCountEndToEnd() {
        QuantumEngine.Engine eng =
            QuantumEngine.createEngine(64L * 1024 * 1024);
        try {
            // Source data
            long[] ks = {1, 2, 1, 3, 2};
            long[] vs = {10, 20, 30, 40, 50};

            QuantumEngine.Column kCol = QuantumEngine.columnInt64(eng, ks, null);
            QuantumEngine.Column vCol = QuantumEngine.columnInt64(eng, vs, null);
            QuantumEngine.Batch batch =
                QuantumEngine.batchNew(eng, new QuantumEngine.Column[]{kCol, vCol});
            QuantumEngine.registerSource(eng, "kv",
                new QuantumEngine.Batch[]{batch});

            // Build PlanNode IR via Java protobuf.
            QuantumPlan.Schema schema = QuantumPlan.Schema.newBuilder()
                .addFields(QuantumPlan.Field.newBuilder()
                    .setName("k").setType(QuantumPlan.TypeId.TYPE_INT64))
                .addFields(QuantumPlan.Field.newBuilder()
                    .setName("v").setType(QuantumPlan.TypeId.TYPE_INT64))
                .build();
            QuantumPlan.PlanNode scan = QuantumPlan.PlanNode.newBuilder()
                .setScan(QuantumPlan.ScanNode.newBuilder()
                    .setSourceId("kv").setSchema(schema))
                .build();
            QuantumPlan.ScalarExpr colK = QuantumPlan.ScalarExpr.newBuilder()
                .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(0))
                .build();
            QuantumPlan.ScalarExpr colV = QuantumPlan.ScalarExpr.newBuilder()
                .setColumnRef(QuantumPlan.ColumnRef.newBuilder().setIndex(1))
                .build();
            QuantumPlan.AggregateSpec sumAgg = QuantumPlan.AggregateSpec.newBuilder()
                .setKind(QuantumPlan.AggregateKind.AGG_SUM)
                .setInputType(QuantumPlan.TypeId.TYPE_INT64)
                .setInput(colV)
                .build();
            QuantumPlan.AggregateSpec cntAgg = QuantumPlan.AggregateSpec.newBuilder()
                .setKind(QuantumPlan.AggregateKind.AGG_COUNT_STAR)
                .build();
            QuantumPlan.PlanNode root = QuantumPlan.PlanNode.newBuilder()
                .setHashAgg(QuantumPlan.HashAggregateNode.newBuilder()
                    .setChild(scan)
                    .addGroupKeys(colK)
                    .addAggregates(sumAgg)
                    .addAggregates(cntAgg)
                    .addOutputNames("k").addOutputNames("sum_v").addOutputNames("cnt"))
                .build();

            ByteString bytes = root.toByteString();
            QuantumEngine.Result res = QuantumEngine.execute(eng, bytes.toByteArray());
            try {
                Map<Long, long[]> got = new HashMap<>();
                QuantumEngine.Batch out;
                while ((out = QuantumEngine.resultNext(res)) != null) {
                    long n = QuantumEngine.batchNumRows(out);
                    assertEquals(3, QuantumEngine.batchNumCols(out));
                    assertEquals(QuantumEngine.Q_TYPE_INT64,
                                 QuantumEngine.batchColType(out, 0));
                    long[] kArr = QuantumEngine.batchColInt64(out, 0, n);
                    long[] sArr = QuantumEngine.batchColInt64(out, 1, n);
                    long[] cArr = QuantumEngine.batchColInt64(out, 2, n);
                    for (int i = 0; i < n; ++i) {
                        got.put(kArr[i], new long[]{sArr[i], cArr[i]});
                    }
                }
                assertArrayEquals(new long[]{40, 2}, got.get(1L));
                assertArrayEquals(new long[]{70, 2}, got.get(2L));
                assertArrayEquals(new long[]{40, 1}, got.get(3L));
            } finally {
                QuantumEngine.destroyResult(res);
            }
        } finally {
            QuantumEngine.destroyEngine(eng);
        }
    }

    /** Malformed protobuf bytes should surface a useful error string. */
    @Test
    void badPlanBytesYieldsErrorMessage() {
        QuantumEngine.Engine eng = QuantumEngine.createEngine(4096);
        try {
            byte[] garbage = {(byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF};
            RuntimeException ex = assertThrows(RuntimeException.class,
                () -> QuantumEngine.execute(eng, garbage));
            assertTrue(ex.getMessage().contains("PlanProto"),
                       "expected error mentioning PlanProto, got: " + ex.getMessage());
        } finally {
            QuantumEngine.destroyEngine(eng);
        }
    }
}
