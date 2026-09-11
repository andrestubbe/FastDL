package fastdl.tensor;

import fastmemory.Memory;
import fastpointer.Pointer;

import java.util.Arrays;

/**
 * GC-free, 32-byte SIMD-aligned multidimensional float tensor backed by native off-heap memory.
 *
 * <p>This is the performance-critical counterpart to the standard heap-based {@link Tensor}.
 * Instead of a Java {@code float[]} on the JVM heap, the data lives in native memory
 * allocated by {@code fastmemory.Memory} — completely invisible to the Garbage Collector.
 * This eliminates GC pauses during large training batches and guarantees the 32-byte
 * boundary alignment required for AVX2 SIMD instructions.
 *
 * <h2>Why off-heap?</h2>
 * <ul>
 *   <li><b>No GC pauses</b> — the JVM never has to scan, move, or collect this memory.
 *       Critical for long training runs where stop-the-world GC events would interrupt
 *       the gradient descent loop.</li>
 *   <li><b>32-byte alignment</b> — AVX2 loads/stores ({@code _mm256_load_ps}) require
 *       the base address to be aligned on 32-byte boundaries.
 *       {@code Memory.allocateAligned(n, 32)} guarantees this, enabling the full
 *       throughput of {@code VectorTensorOps}.</li>
 *   <li><b>OS page locking</b> — via {@code Memory.lockPages()}, the OS can be told
 *       never to swap this memory to disk, eliminating random latency spikes.</li>
 * </ul>
 *
 * <h2>Memory layout</h2>
 * Data is stored in <b>row-major (C-contiguous)</b> order as a flat array of 32-bit
 * IEEE 754 floats. Multidimensional indexing uses pre-computed strides:
 * <pre>
 *   element at indices [i₀, i₁, ..., iₙ]  =  base + Σ(iₖ × strideₖ) × 4 bytes
 * </pre>
 *
 * <h2>Lifecycle — MUST call free()</h2>
 * Unlike heap arrays, native memory is <b>not automatically reclaimed</b> when the object
 * becomes unreachable. Always call {@link #free()} when done, ideally in a
 * {@code try-finally} block:
 * <pre>{@code
 * OffHeapTensor t = new OffHeapTensor(128, 64);
 * try {
 *     // ... use t ...
 * } finally {
 *     t.free();
 * }
 * }</pre>
 *
 * <h2>Conversion to/from heap</h2>
 * <ul>
 *   <li>{@link #toArray()} — copies data back to a Java {@code float[]} (heap)</li>
 *   <li>{@link #setAll(float[])} — bulk-writes from a heap array into native memory</li>
 * </ul>
 *
 * <h2>FastJava dependencies</h2>
 * Uses {@code fastmemory.Memory} (FastMemory 0.1.1) and {@code fastpointer.Pointer}
 * (FastPointer 0.1.1) directly — no intermediate wrapper layer.
 *
 * @see fastdl.tensor.Tensor
 * @see fastdl.ops.VectorTensorOps
 */
public final class OffHeapTensor {
    private final Memory  memory;
    private final Pointer pointer;
    private final int[] shape;
    private final int[] strides;
    private final int total;

    public OffHeapTensor(int... shape) {
        this.shape = shape != null && shape.length > 0 ? shape.clone() : new int[] {1};
        this.strides = computeStrides(this.shape);
        this.total = computeSize(this.shape);
        this.memory  = Memory.allocateAligned(this.total * Float.BYTES, 32);
        this.pointer = this.memory.pointer();
    }

    public OffHeapTensor(float[] values, int... shape) {
        this.shape = shape != null && shape.length > 0 ? shape.clone() : new int[] {values.length};
        this.strides = computeStrides(this.shape);
        this.total = computeSize(this.shape);
        this.memory  = Memory.allocateAligned(this.total * Float.BYTES, 32);
        this.pointer = this.memory.pointer();
        setAll(values);
    }

    public int[] shape() {
        return shape.clone();
    }

    public int size() {
        return total;
    }

    public float get(int... indices) {
        return pointer.getFloat((long) offset(indices) * Float.BYTES);
    }

    public void set(float value, int... indices) {
        pointer.setFloat((long) offset(indices) * Float.BYTES, value);
    }

    public void setAll(float[] values) {
        if (values.length != total) {
            throw new IllegalArgumentException("Expected " + total + " values but got " + values.length);
        }
        for (int i = 0; i < values.length; i++) {
            pointer.setFloat((long) i * Float.BYTES, values[i]);
        }
    }

    public float[] toArray() {
        float[] out = new float[total];
        for (int i = 0; i < total; i++) {
            out[i] = pointer.getFloat((long) i * Float.BYTES);
        }
        return out;
    }

    public void free() {
        memory.free();
    }

    private int offset(int... indices) {
        int off = 0;
        for (int i = 0; i < indices.length; i++) {
            off += indices[i] * strides[i];
        }
        return off;
    }

    private static int[] computeStrides(int[] shape) {
        int[] s = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = stride;
            stride *= shape[i];
        }
        return s;
    }

    private static int computeSize(int[] shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        return size;
    }

    @Override
    public String toString() {
        float[] values = toArray();
        return "OffHeapTensor(shape=" + Arrays.toString(shape) + ", data="
            + (values.length <= 8 ? Arrays.toString(values) : "[" + values[0] + ", ..., " + values[values.length - 1] + "]")
            + ")";
    }
}
