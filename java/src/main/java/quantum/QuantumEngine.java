package quantum;

import com.sun.jna.Library;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.PointerType;
import com.sun.jna.ptr.PointerByReference;

/**
 * JNA bindings to the Quantum Engine C ABI declared in
 * {@code include/quantum/quantum_c.h}.
 *
 * <p>Each opaque handle ({@link Engine}, {@link Column}, {@link Batch},
 * {@link Result}) wraps a {@link Pointer}. Lifecycle: callers create an
 * Engine, build Columns + Batches against it, register sources, execute
 * a serialized protobuf plan, then iterate Result batches.
 *
 * <p>Errors are surfaced through {@link #lastError()} which reflects the
 * thread-local error string set by the most recent C call.
 */
public final class QuantumEngine {

    /** Strongly-typed opaque handle for the native engine. */
    public static final class Engine extends PointerType {
        public Engine() {}
        public Engine(Pointer p) { super(p); }
    }
    public static final class Column extends PointerType {
        public Column() {}
        public Column(Pointer p) { super(p); }
    }
    public static final class Batch extends PointerType {
        public Batch() {}
        public Batch(Pointer p) { super(p); }
    }
    public static final class Result extends PointerType {
        public Result() {}
        public Result(Pointer p) { super(p); }
    }

    // Column type codes — mirror q_type_t in quantum_c.h.
    public static final int Q_TYPE_BOOL    = 0;
    public static final int Q_TYPE_INT32   = 1;
    public static final int Q_TYPE_INT64   = 2;
    public static final int Q_TYPE_FLOAT64 = 3;
    public static final int Q_TYPE_STRING  = 4;

    /** Native binding interface — JNA generates the dispatchers. */
    interface Native_ extends Library {
        Engine     q_create_engine(long budget_bytes);
        void       q_destroy_engine(Engine engine);
        String     q_last_error();

        Column     q_column_int64(Engine engine, Pointer data, Pointer validity,
                                  long num_rows);
        Column     q_column_string(Engine engine, Pointer values, Pointer lens,
                                   Pointer validity, long num_rows);

        Batch      q_batch_new(Engine engine, Pointer cols, long num_cols);
        void       q_register_source(Engine engine, String source_id,
                                     Pointer batches, long num_batches);

        Result     q_execute(Engine engine, Pointer plan_bytes, long plan_len);
        void       q_destroy_result(Result result);
        Batch      q_result_next(Result result);

        long       q_batch_num_rows(Batch b);
        long       q_batch_num_cols(Batch b);
        int        q_batch_col_type(Batch b, long col);
        Pointer    q_batch_col_int64(Batch b, long col);
        Pointer    q_batch_col_float64(Batch b, long col);
        int        q_batch_col_valid(Batch b, long col, long row);
        long       q_batch_col_string_at(Batch b, long col, long row,
                                         PointerByReference out_ptr);
    }

    public static final Native_ NATIVE =
        Native.load("quantum_core", Native_.class);

    public static String lastError() { return NATIVE.q_last_error(); }

    public static Engine createEngine(long budgetBytes) {
        Engine e = NATIVE.q_create_engine(budgetBytes);
        if (e == null) {
            throw new RuntimeException("q_create_engine failed: " + lastError());
        }
        return e;
    }

    public static void destroyEngine(Engine e) { NATIVE.q_destroy_engine(e); }

    /** Build an int64 column from a Java long[] (validity = null for all-valid). */
    public static Column columnInt64(Engine e, long[] values, byte[] validity) {
        Memory dataMem = new Memory((long) values.length * Long.BYTES);
        dataMem.write(0, values, 0, values.length);

        Memory validityMem = null;
        if (validity != null) {
            validityMem = new Memory(validity.length);
            validityMem.write(0, validity, 0, validity.length);
        }
        Column col = NATIVE.q_column_int64(e, dataMem,
            validityMem == null ? Pointer.NULL : validityMem, values.length);
        if (col == null) {
            throw new RuntimeException("q_column_int64 failed: " + lastError());
        }
        return col;
    }

    /** Build a batch from an array of engine-owned columns (consumed). */
    public static Batch batchNew(Engine e, Column[] cols) {
        Memory mem = new Memory((long) cols.length * Native.POINTER_SIZE);
        for (int i = 0; i < cols.length; ++i) {
            mem.setPointer((long) i * Native.POINTER_SIZE, cols[i].getPointer());
        }
        Batch b = NATIVE.q_batch_new(e, mem, cols.length);
        if (b == null) {
            throw new RuntimeException("q_batch_new failed: " + lastError());
        }
        return b;
    }

    /** Register a vector of engine-owned batches (consumed) under source_id. */
    public static void registerSource(Engine e, String sourceId, Batch[] batches) {
        Memory mem = new Memory((long) batches.length * Native.POINTER_SIZE);
        for (int i = 0; i < batches.length; ++i) {
            mem.setPointer((long) i * Native.POINTER_SIZE, batches[i].getPointer());
        }
        NATIVE.q_register_source(e, sourceId, mem, batches.length);
    }

    /** Execute a serialized protobuf PlanNode. */
    public static Result execute(Engine e, byte[] planBytes) {
        Memory mem = new Memory(planBytes.length);
        mem.write(0, planBytes, 0, planBytes.length);
        Result r = NATIVE.q_execute(e, mem, planBytes.length);
        if (r == null) {
            throw new RuntimeException("q_execute failed: " + lastError());
        }
        return r;
    }

    public static Batch resultNext(Result r) { return NATIVE.q_result_next(r); }
    public static void  destroyResult(Result r) { NATIVE.q_destroy_result(r); }

    public static long   batchNumRows(Batch b) { return NATIVE.q_batch_num_rows(b); }
    public static long   batchNumCols(Batch b) { return NATIVE.q_batch_num_cols(b); }
    public static int    batchColType(Batch b, long c) { return NATIVE.q_batch_col_type(b, c); }
    public static long[] batchColInt64(Batch b, long c, long n) {
        Pointer p = NATIVE.q_batch_col_int64(b, c);
        return p.getLongArray(0, (int) n);
    }
    public static double[] batchColFloat64(Batch b, long c, long n) {
        Pointer p = NATIVE.q_batch_col_float64(b, c);
        return p.getDoubleArray(0, (int) n);
    }
    public static boolean batchColValid(Batch b, long c, long r) {
        return NATIVE.q_batch_col_valid(b, c, r) != 0;
    }
    public static String batchColStringAt(Batch b, long c, long r) {
        PointerByReference ref = new PointerByReference();
        long len = NATIVE.q_batch_col_string_at(b, c, r, ref);
        if (len < 0) return null;
        if (len == 0) return "";
        return new String(ref.getValue().getByteArray(0, (int) len),
                          java.nio.charset.StandardCharsets.UTF_8);
    }

    private QuantumEngine() {}
}
