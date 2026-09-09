package fastdl.ops;

import fastdl.tensor.Tensor;

/**
 * Core mathematical operations on {@link fastdl.tensor.Tensor} objects used throughout
 * the FastDL neural network stack.
 *
 * <p>All methods are <strong>stateless and static</strong>. There is no automatic
 * differentiation here — each calling layer is responsible for implementing its own
 * {@code backward()} pass using the intermediate values it cached during {@code forward()}.
 *
 * <h2>Operations provided</h2>
 * <ul>
 *   <li><b>matmul</b> — 2-D and 3-D batched matrix multiplication [M,K] × [K,N] → [M,N]</li>
 *   <li><b>batchedMatmul</b> — fully batched [B,M,K] × [B,K,N] → [B,M,N]</li>
 *   <li><b>softmax</b> — numerically stable row-wise softmax along the last dimension</li>
 *   <li><b>layerNorm</b> — Layer Normalization over the last dimension with learnable affine parameters</li>
 *   <li><b>gelu</b> — GELU activation used in Transformer feedforward blocks</li>
 *   <li><b>geluBackward</b> — analytic GELU derivative for the backward pass</li>
 *   <li><b>add</b> — element-wise addition with automatic broadcast of smaller tensors</li>
 *   <li><b>transpose</b> — swaps the last two dimensions of any rank tensor</li>
 *   <li><b>scale</b> — multiplies every element by a scalar factor</li>
 *   <li><b>reshape</b> — reinterprets the flat buffer with new shape (zero-copy)</li>
 * </ul>
 *
 * <h2>Performance notes</h2>
 * All loops use plain Java scalar arithmetic. For production workloads, replace the
 * inner matmul loops with FastSIMD AVX2 intrinsics or FastGPU Vulkan Compute kernels
 * (see the acceleration roadmap in {@code DOING.md}).
 *
 * <h2>Usage example</h2>
 * <pre>{@code
 * Tensor a   = Tensor.randn(4, 64);   // [batch=4, dModel=64]
 * Tensor b   = Tensor.randn(64, 128); // [dModel=64, outDim=128]
 * Tensor out = TensorOps.matmul(a, b); // [4, 128]
 * Tensor act = TensorOps.gelu(out);
 * }</pre>
 *
 * @see fastdl.layer.Dense
 * @see fastdl.model.MultiHeadAttention
 * @see fastdl.layer.LayerNorm
 */
public final class TensorOps {

    private TensorOps() {}

    // -------------------------------------------------------------------------
    // Matrix multiply: [batch, M, K] x [K, N] -> [batch, M, N]
    // Also handles 2-D: [M, K] x [K, N] -> [M, N]
    // -------------------------------------------------------------------------
    public static Tensor matmul(Tensor a, Tensor b) {
        return VectorTensorOps.matmul(a, b);
    }

    // -------------------------------------------------------------------------
    // Batched matmul: [B, M, K] x [B, K, N] -> [B, M, N]
    // -------------------------------------------------------------------------
    public static Tensor batchedMatmul(Tensor a, Tensor b) {
        return VectorTensorOps.batchedMatmul(a, b);
    }

    // -------------------------------------------------------------------------
    // Softmax along last dimension. Input shape: [*, D]
    // -------------------------------------------------------------------------
    public static Tensor softmax(Tensor x) {
        int[] shape = x.shape();
        int last = shape[shape.length - 1];
        int rows = x.size() / last;
        Tensor out = new Tensor(shape);
        float[] xd = x.data();
        float[] od = out.data();
        for (int r = 0; r < rows; r++) {
            int base = r * last;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < last; i++) {
                if (xd[base + i] > max) max = xd[base + i];
            }
            float sum = 0f;
            for (int i = 0; i < last; i++) {
                od[base + i] = (float) Math.exp(xd[base + i] - max);
                sum += od[base + i];
            }
            for (int i = 0; i < last; i++) {
                od[base + i] /= sum;
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // LayerNorm over last dimension: (x - mean) / sqrt(var + eps) * weight + bias
    // shape: [*, D]; weight and bias: [D]
    // -------------------------------------------------------------------------
    public static Tensor layerNorm(Tensor x, Tensor weight, Tensor bias, float eps) {
        int[] shape = x.shape();
        int D = shape[shape.length - 1];
        int rows = x.size() / D;
        Tensor out = new Tensor(shape);
        float[] xd = x.data();
        float[] wd = weight.data();
        float[] bd = bias.data();
        float[] od = out.data();
        for (int r = 0; r < rows; r++) {
            int base = r * D;
            double mean = 0.0;
            for (int i = 0; i < D; i++) mean += xd[base + i];
            mean /= D;
            double var = 0.0;
            for (int i = 0; i < D; i++) {
                double diff = xd[base + i] - mean;
                var += diff * diff;
            }
            var = var / D + eps;
            float invStd = (float) (1.0 / Math.sqrt(var));
            for (int i = 0; i < D; i++) {
                od[base + i] = (xd[base + i] - (float) mean) * invStd * wd[i] + bd[i];
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // -------------------------------------------------------------------------
    private static final float SQRT_2_OVER_PI = 0.7978845608f;
    private static final float GELU_COEF = 0.044715f;

    public static Tensor gelu(Tensor x) {
        float[] xd = x.data();
        Tensor out = new Tensor(x.shape());
        float[] od = out.data();
        for (int i = 0; i < xd.length; i++) {
            float v = xd[i];
            float inner = SQRT_2_OVER_PI * (v + GELU_COEF * v * v * v);
            od[i] = 0.5f * v * (1f + (float) Math.tanh(inner));
        }
        return out;
    }

    // GELU derivative (for backward pass)
    public static Tensor geluBackward(Tensor x, Tensor gradOut) {
        float[] xd = x.data();
        float[] go = gradOut.data();
        Tensor grad = new Tensor(x.shape());
        float[] gd = grad.data();
        for (int i = 0; i < xd.length; i++) {
            float v = xd[i];
            float inner = SQRT_2_OVER_PI * (v + GELU_COEF * v * v * v);
            float tanh = (float) Math.tanh(inner);
            float dtanh = 1f - tanh * tanh;
            float dInner = SQRT_2_OVER_PI * (1f + 3f * GELU_COEF * v * v);
            float dGelu = 0.5f * (1f + tanh) + 0.5f * v * dtanh * dInner;
            gd[i] = go[i] * dGelu;
        }
        return grad;
    }

    // -------------------------------------------------------------------------
    // Element-wise add — supports broadcasting bias [D] onto [B, T, D]
    // -------------------------------------------------------------------------
    public static Tensor add(Tensor a, Tensor b) {
        if (a.size() == b.size()) {
            Tensor out = new Tensor(a.shape());
            float[] ad = a.data(), bd = b.data(), od = out.data();
            for (int i = 0; i < ad.length; i++) od[i] = ad[i] + bd[i];
            return out;
        }
        // broadcast b (smaller) over a (larger)
        Tensor out = new Tensor(a.shape());
        float[] ad = a.data(), bd = b.data(), od = out.data();
        int bSize = b.size();
        for (int i = 0; i < ad.length; i++) {
            od[i] = ad[i] + bd[i % bSize];
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Transpose last two dims: [B, M, N] -> [B, N, M]
    // -------------------------------------------------------------------------
    public static Tensor transpose(Tensor x) {
        int[] shape = x.shape();
        int rank = shape.length;
        int M = shape[rank - 2];
        int N = shape[rank - 1];
        int batch = x.size() / (M * N);
        int[] newShape = shape.clone();
        newShape[rank - 2] = N;
        newShape[rank - 1] = M;
        Tensor out = new Tensor(newShape);
        float[] xd = x.data();
        float[] od = out.data();
        for (int b = 0; b < batch; b++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    od[b * N * M + n * M + m] = xd[b * M * N + m * N + n];
                }
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Scale: multiply all elements by scalar
    // -------------------------------------------------------------------------
    public static Tensor scale(Tensor x, float factor) {
        Tensor out = new Tensor(x.shape());
        float[] xd = x.data(), od = out.data();
        for (int i = 0; i < xd.length; i++) od[i] = xd[i] * factor;
        return out;
    }

    // -------------------------------------------------------------------------
    // Reshape — no data copy, just reinterpret (flat layout must match)
    // -------------------------------------------------------------------------
    public static Tensor reshape(Tensor x, int... newShape) {
        return new Tensor(x.data(), newShape);
    }
}
